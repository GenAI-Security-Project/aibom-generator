#!/usr/bin/env python
import os
import json
import logging
import sys
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime
from datasets import Dataset, load_dataset, concatenate_datasets
from typing import Dict, Optional, Any, List
import uuid
import re # Import regex module
import html # Import html module for escaping
from urllib.parse import urlparse
from starlette.middleware.base import BaseHTTPMiddleware
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError # For specific error handling

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Registry-driven field classification imports
try:
    from src.aibom_generator.field_registry_manager import (
        get_field_registry_manager,
        generate_field_classification,
        get_configurable_scoring_weights
    )
    REGISTRY_MANAGER = get_field_registry_manager()
    FIELD_CLASSIFICATION = generate_field_classification()
    SCORING_WEIGHTS = get_configurable_scoring_weights()
    REGISTRY_AVAILABLE = True
    logger.info(f"✅ Registry-driven API: {len(FIELD_CLASSIFICATION)} fields loaded")
except ImportError as e:
    REGISTRY_AVAILABLE = False
    FIELD_CLASSIFICATION = {}
    SCORING_WEIGHTS = {}
    logger.warning(f"⚠️ Registry not available for API: {e}")

# Define directories and constants
templates_dir = "templates"
OUTPUT_DIR = "/tmp/aibom_output"
MAX_AGE_DAYS = 7  # Remove files older than 7 days
MAX_FILES = 1000  # Keep maximum 1000 files
CLEANUP_INTERVAL = 100  # Run cleanup every 100 requests

# --- Add Counter Configuration (started as of May 3, 2025) ---
HF_REPO = "aetheris-ai/aisbom-usage-log"  # User needs to create this private repo
HF_TOKEN = os.getenv("HF_TOKEN")  # User must set this environment variable
# --- End Counter Configuration ---

# Create app
app = FastAPI(title="AI SBOM Generator API")

# Try different import paths
try:
    from src.aibom_generator.rate_limiting import RateLimitMiddleware, ConcurrencyLimitMiddleware, RequestSizeLimitMiddleware
    logger.info("Successfully imported rate_limiting from src.aibom_generator")
except ImportError:
    try:
        from .rate_limiting import RateLimitMiddleware, ConcurrencyLimitMiddleware, RequestSizeLimitMiddleware
        logger.info("Successfully imported rate_limiting with relative import")
    except ImportError:
        try:
            from rate_limiting import RateLimitMiddleware, ConcurrencyLimitMiddleware, RequestSizeLimitMiddleware
            logger.info("Successfully imported rate_limiting from current directory")
        except ImportError:
            logger.error("Could not import rate_limiting, DoS protection disabled")
            # Define dummy middleware classes that just pass through requests
            class RateLimitMiddleware(BaseHTTPMiddleware):
                def __init__(self, app, **kwargs):
                    super().__init__(app)
                async def dispatch(self, request, call_next):
                    try:
                        return await call_next(request)
                    except Exception as e:
                        logger.error(f"Error in RateLimitMiddleware: {str(e)}")
                        return JSONResponse(
                            status_code=500,
                            content={"detail": f"Internal server error: {str(e)}"}
                        )
                        
            class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
                def __init__(self, app, **kwargs):
                    super().__init__(app)
                async def dispatch(self, request, call_next):
                    try:
                        return await call_next(request)
                    except Exception as e:
                        logger.error(f"Error in ConcurrencyLimitMiddleware: {str(e)}")
                        return JSONResponse(
                            status_code=500,
                            content={"detail": f"Internal server error: {str(e)}"}
                        )
                        
            class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
                def __init__(self, app, **kwargs):
                    super().__init__(app)
                async def dispatch(self, request, call_next):
                    try:
                        return await call_next(request)
                    except Exception as e:
                        logger.error(f"Error in RequestSizeLimitMiddleware: {str(e)}")
                        return JSONResponse(
                            status_code=500,
                            content={"detail": f"Internal server error: {str(e)}"}
                        )
try:
    from src.aibom_generator.captcha import verify_recaptcha
    logger.info("Successfully imported captcha from src.aibom_generator")
except ImportError:
    try:
        from .captcha import verify_recaptcha
        logger.info("Successfully imported captcha with relative import")
    except ImportError:
        try:
            from captcha import verify_recaptcha
            logger.info("Successfully imported captcha from current directory")
        except ImportError:
            logger.warning("Could not import captcha module, CAPTCHA verification disabled")
            # Define a dummy verify_recaptcha function that always succeeds
            def verify_recaptcha(response_token=None):
                logger.warning("Using dummy CAPTCHA verification (always succeeds)")
                return True



# Rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    rate_limit_per_minute=10,  # Adjust as needed
    rate_limit_window=60,
    protected_routes=["/generate", "/api/generate", "/api/generate-with-report"]
)

app.add_middleware(
    ConcurrencyLimitMiddleware,
    max_concurrent_requests=5,  # Adjust based on server capacity
    timeout=5.0,
    protected_routes=["/generate", "/api/generate", "/api/generate-with-report"]
)


# Size limiting middleware
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_content_length=1024*1024  # 1MB
)


# Define models
class StatusResponse(BaseModel):
    status: str
    version: str
    generator_version: str

# Initialize templates
templates = Jinja2Templates(directory=templates_dir)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount output directory as static files
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Request counter for periodic cleanup
request_counter = 0

# Import cleanup_utils using absolute import
try:
    from src.aibom_generator.cleanup_utils import perform_cleanup
    logger.info("Successfully imported cleanup_utils")
except ImportError:
    try:
        from cleanup_utils import perform_cleanup
        logger.info("Successfully imported cleanup_utils from current directory")
    except ImportError:
        logger.error("Could not import cleanup_utils, defining functions inline")
        # Define cleanup functions inline if import fails
        def cleanup_old_files(directory, max_age_days=7):
            """Remove files older than max_age_days from the specified directory."""
            if not os.path.exists(directory):
                logger.warning(f"Directory does not exist: {directory}")
                return 0
            
            removed_count = 0
            now = datetime.now()
            
            try:
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        file_age = now - datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_age.days > max_age_days:
                            try:
                                os.remove(file_path)
                                removed_count += 1
                                logger.info(f"Removed old file: {file_path}")
                            except Exception as e:
                                logger.error(f"Error removing file {file_path}: {e}")
                
                logger.info(f"Cleanup completed: removed {removed_count} files older than {max_age_days} days from {directory}")
                return removed_count
            except Exception as e:
                logger.error(f"Error during cleanup of directory {directory}: {e}")
                return 0

        def limit_file_count(directory, max_files=1000):
            """Ensure no more than max_files are kept in the directory (removes oldest first)."""
            if not os.path.exists(directory):
                logger.warning(f"Directory does not exist: {directory}")
                return 0
            
            removed_count = 0
            
            try:
                files = []
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        files.append((file_path, os.path.getmtime(file_path)))
                
                # Sort by modification time (oldest first)
                files.sort(key=lambda x: x[1])
                
                # Remove oldest files if limit is exceeded
                files_to_remove = files[:-max_files] if len(files) > max_files else []
                
                for file_path, _ in files_to_remove:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        logger.info(f"Removed excess file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {e}")
                
                logger.info(f"File count limit enforced: removed {removed_count} oldest files from {directory}, keeping max {max_files}")
                return removed_count
            except Exception as e:
                logger.error(f"Error during file count limiting in directory {directory}: {e}")
                return 0

        def perform_cleanup(directory, max_age_days=7, max_files=1000):
            """Perform both time-based and count-based cleanup."""
            time_removed = cleanup_old_files(directory, max_age_days)
            count_removed = limit_file_count(directory, max_files)
            return time_removed + count_removed

# Run initial cleanup
try:
    removed = perform_cleanup(OUTPUT_DIR, MAX_AGE_DAYS, MAX_FILES)
    logger.info(f"Initial cleanup removed {removed} files")
except Exception as e:
    logger.error(f"Error during initial cleanup: {e}")

# Define middleware
@app.middleware("http" )
async def cleanup_middleware(request, call_next):
    """Middleware to periodically run cleanup."""
    global request_counter
    
    # Increment request counter
    request_counter += 1
    
    # Run cleanup periodically
    if request_counter % CLEANUP_INTERVAL == 0:
        logger.info(f"Running scheduled cleanup after {request_counter} requests")
        try:
            removed = perform_cleanup(OUTPUT_DIR, MAX_AGE_DAYS, MAX_FILES)
            logger.info(f"Scheduled cleanup removed {removed} files")
        except Exception as e:
            logger.error(f"Error during scheduled cleanup: {e}")
    
    # Process the request
    response = await call_next(request)
    return response


# --- Model ID Validation and Normalization Helpers --- 
# Regex for valid Hugging Face ID parts (alphanumeric, -, _, .)
# Allows owner/model format
HF_ID_REGEX = re.compile(r"^[a-zA-Z0-9\.\-\_]+/[a-zA-Z0-9\.\-\_]+$")

def is_valid_hf_input(input_str: str) -> bool:
    """Checks if the input is a valid Hugging Face model ID or URL."""
    if not input_str or len(input_str) > 200: # Basic length check
        return False
        
    if input_str.startswith(("http://", "https://") ):
        try:
            parsed = urlparse(input_str)
            # Check domain and path structure
            if parsed.netloc == "huggingface.co":
                path_parts = parsed.path.strip("/").split("/")
                # Must have at least owner/model, can have more like /tree/main
                if len(path_parts) >= 2 and path_parts[0] and path_parts[1]:
                     # Check characters in the relevant parts
                     if re.match(r"^[a-zA-Z0-9\.\-\_]+$", path_parts[0]) and \
                        re.match(r"^[a-zA-Z0-9\.\-\_]+$", path_parts[1]):
                         return True
            return False # Not a valid HF URL format
        except Exception:
            return False # URL parsing failed
    else:
        # Assume owner/model format, check with regex
        return bool(HF_ID_REGEX.match(input_str))

def _normalise_model_id(raw_id: str) -> str:
    """
    Accept either validated 'owner/model' or a validated full URL like
    'https://huggingface.co/owner/model'. Return 'owner/model'.
    Assumes input has already been validated by is_valid_hf_input.
    """
    if raw_id.startswith(("http://", "https://") ):
        path = urlparse(raw_id).path.lstrip("/")
        parts = path.split("/")
        # We know from validation that parts[0] and parts[1] exist
        return f"{parts[0]}/{parts[1]}"
    return raw_id # Already in owner/model format

# --- End Model ID Helpers ---


# --- Add Counter Helper Functions ---
def log_sbom_generation(model_id: str):
    """Logs a successful SBOM generation event to the Hugging Face dataset."""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set. Skipping SBOM generation logging.")
        return

    try:
        # Normalize model_id before logging
        normalized_model_id_for_log = _normalise_model_id(model_id) # added to normalize id
        log_data = {
            "timestamp": [datetime.utcnow().isoformat()],
            "event": ["generated"],
            "model_id": [normalized_model_id_for_log] # use normalized_model_id_for_log
        }
        ds_new_log = Dataset.from_dict(log_data)

        # Try to load existing dataset to append
        try:
            # Use trust_remote_code=True if required by the dataset/model on HF
            # Corrected: Removed unnecessary backslashes around 'train'
            existing_ds = load_dataset(HF_REPO, token=HF_TOKEN, split='train', trust_remote_code=True)
            # Check if dataset is empty or has different columns (handle initial creation)
            if len(existing_ds) > 0 and set(existing_ds.column_names) == set(log_data.keys()):
                 ds_to_push = concatenate_datasets([existing_ds, ds_new_log])
            elif len(existing_ds) == 0:
                 logger.info(f"Dataset {HF_REPO} is empty. Pushing initial data.")
                 ds_to_push = ds_new_log
            else:
                 logger.warning(f"Dataset {HF_REPO} has unexpected columns {existing_ds.column_names} vs {list(log_data.keys())}. Appending new log anyway, structure might differ.")
                 # Attempt concatenation even if columns differ slightly, HF might handle it
                 # Or consider more robust schema migration/handling if needed
                 ds_to_push = concatenate_datasets([existing_ds, ds_new_log])

        except Exception as load_err:
             # Handle case where dataset doesn't exist yet or other loading errors
             # Corrected: Removed unnecessary backslash in doesn't
             logger.info(f"Could not load existing dataset {HF_REPO} (may not exist yet): {load_err}. Pushing new dataset.")
             ds_to_push = ds_new_log # ds is already prepared with the new log entry

        # Push the updated or new dataset
        # Corrected: Removed unnecessary backslash in it's
        ds_to_push.push_to_hub(HF_REPO, token=HF_TOKEN, private=True) # Ensure it's private
        logger.info(f"Successfully logged SBOM generation for {normalized_model_id_for_log} to {HF_REPO}") # use normalized model id

    except Exception as e:
        logger.error(f"Failed to log SBOM generation to {HF_REPO}: {e}")

def get_sbom_count() -> str:
    """Retrieves the total count of generated SBOMs from the Hugging Face dataset."""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set. Cannot retrieve SBOM count.")
        return "N/A"
    try:
        # Load the dataset - assumes 'train' split exists after first push
        # Use trust_remote_code=True if required by the dataset/model on HF
        # Corrected: Removed unnecessary backslashes around 'train'
        ds = load_dataset(HF_REPO, token=HF_TOKEN, split='train', trust_remote_code=True)
        count = len(ds)
        logger.info(f"Retrieved SBOM count: {count} from {HF_REPO}")
        # Format count for display (e.g., add commas for large numbers)
        return f"{count:,}"
    except Exception as e:
        logger.error(f"Failed to retrieve SBOM count from {HF_REPO}: {e}")
        # Return "N/A" or similar indicator on error
        return "N/A"
# --- End Counter Helper Functions ---

@app.on_event("startup")
async def startup_event():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory ready at {OUTPUT_DIR}")
    logger.info(f"Registered routes: {[route.path for route in app.routes]}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    sbom_count = get_sbom_count() # Get count
    try:
        return templates.TemplateResponse("index.html", {"request": request, "sbom_count": sbom_count}) # Pass to template
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        # Attempt to render error page even if main page fails
        try:
            return templates.TemplateResponse("error.html", {"request": request, "error": f"Template rendering error: {str(e)}", "sbom_count": sbom_count})
        except Exception as template_err:
             # Fallback if error template also fails
             logger.error(f"Error rendering error template: {template_err}")
             raise HTTPException(status_code=500, detail=f"Template rendering error: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(status="operational", version="1.0.0", generator_version="1.0.0")

# Import utils module for completeness score calculation
def import_utils():
    """Import utils module with fallback paths."""
    try:
        # Try different import paths
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Try direct import first
        try:
            from utils import calculate_completeness_score
            logger.info("Imported utils.calculate_completeness_score directly")
            return calculate_completeness_score
        except ImportError:
            pass

        # Try from src
        try:
            from src.aibom_generator.utils import calculate_completeness_score            
            logger.info("Imported src.aibom_generator.utils.calculate_completeness_score")
            return calculate_completeness_score
        except ImportError:
            pass

        # Try from aibom_generator
        try:
            from aibom_generator.utils import calculate_completeness_score
            logger.info("Imported aibom_generator.utils.calculate_completeness_score")
            return calculate_completeness_score
        except ImportError:
            pass

        # If all imports fail, use the default implementation
        logger.warning("Could not import calculate_completeness_score, using default implementation")
        return None
    except Exception as e:
        logger.error(f"Error importing utils: {str(e)}")
        return None

# Try to import the calculate_completeness_score function
calculate_completeness_score = import_utils()

# Verify registry integration status
if REGISTRY_AVAILABLE:
    logger.info("✅ API fully integrated with registry system")
else:
    logger.warning("⚠️ API using fallback mode - registry not available")
    

def get_tier_points(tier):
    """Get points for a field tier."""
    tier_points = {
        "critical": 4.0,
        "important": 2.0, 
        "supplementary": 1.0
    }
    return tier_points.get(tier, 1.0)

def create_registry_driven_fallback():
    """Create fallback score using registry configuration."""
    if not REGISTRY_AVAILABLE:
        return create_hardcoded_fallback()
        
    categories = {}
    field_checklist = {}
    max_scores = {}
    
    # Get categories and scores from registry
    for field_name, classification in FIELD_CLASSIFICATION.items():
        category = classification["category"]
        tier = classification["tier"]
        
        # Initialize category if not exists
        if category not in categories:
            categories[category] = {"total": 0, "present": 0}
            max_scores[category] = 0
            
        categories[category]["total"] += 1
        max_scores[category] += get_tier_points(tier)
        
        # Add to field checklist with registry-based tier
        tier_stars = {"critical": "★★★", "important": "★★", "supplementary": "★"}
        field_checklist[field_name] = f"n/a {tier_stars.get(tier, '★')}"
    
    return {
        "total_score": 0,
        "section_scores": {cat: 0 for cat in categories.keys()},
        "max_scores": max_scores,
        "field_checklist": field_checklist,
        "category_details": categories
    }

def create_hardcoded_fallback():
    """Fallback to original hardcoded structure when registry unavailable."""
    return {
        "total_score": 0,
        "section_scores": {
            "required_fields": 0,
            "metadata": 0,
            "component_basic": 0,
            "component_model_card": 0,
            "external_references": 0
        },
        "max_scores": {
            "required_fields": 20,
            "metadata": 20,
            "component_basic": 20,
            "component_model_card": 30,
            "external_references": 10
        },
        "field_checklist": {
            "bomFormat": "n/a ★★★",
            "specVersion": "n/a ★★★",
            "serialNumber": "n/a ★★★",
            "version": "n/a ★★★",
            "name": "n/a ★★★",
            "downloadLocation": "n/a ★★★"
        }
    }

# Helper function to create a comprehensive completeness_score with field_checklist
def create_comprehensive_completeness_score(aibom=None):
    """
    Create a comprehensive completeness_score object with all required attributes.
    Uses registry-driven field classification when available.
    """
    # If we have the calculate_completeness_score function and an AIBOM, use it
    if calculate_completeness_score and aibom:
        try:
            return calculate_completeness_score(aibom, validate=True, use_best_practices=True)
        except Exception as e:
            logger.error(f"Error calculating completeness score: {str(e)}")
            # Fall through to registry-driven fallback
    
    # Use registry-driven fallback
    if REGISTRY_AVAILABLE:
        logger.info("Using registry-driven completeness score fallback")
        return create_registry_driven_fallback()
    else:
        logger.warning("Using hardcoded completeness score fallback")
        return create_hardcoded_fallback()


@app.post("/generate", response_class=HTMLResponse)
async def generate_form(
    request: Request,
    model_id: str = Form(...),
    include_inference: bool = Form(False),
    use_best_practices: bool = Form(True),
    g_recaptcha_response: Optional[str] = Form(None)
):
     # Debug log all form data
    form_data = await request.form()
    logger.info(f"All form data: {dict(form_data)}")
    
    # Verify CAPTCHA
    if not verify_recaptcha(g_recaptcha_response):
        return templates.TemplateResponse(
            "error.html", {
                "request": request, 
                "error": "Security verification failed. Please try again.",
                "sbom_count": get_sbom_count()
            }
        )
        
    sbom_count = get_sbom_count() # Get count early for context

    # --- Input Format Validation (BEFORE sanitization for security) ---
    if not is_valid_hf_input(model_id):
        error_message = "Invalid input format. Please provide a valid Hugging Face model ID (e.g., 'owner/model') or a full model URL (e.g., 'https://huggingface.co/owner/model') ."
        logger.warning(f"Invalid model input format received: {model_id}") # Log original input
        # Sanitize for safe display in error response
        sanitized_for_display = html.escape(model_id)
        return templates.TemplateResponse(
            "error.html", {"request": request, "error": error_message, "sbom_count": sbom_count, "model_id": sanitized_for_display}
        )

    # --- Normalize the VALIDATED model ID (use raw input, not HTML-escaped) ---
    # Normalization needs to parse URLs which contain special chars like / and :
    normalized_model_id = _normalise_model_id(model_id)

    # Sanitize for safe display/logging after validation passes
    sanitized_model_id = html.escape(model_id)

    # --- Check if the ID corresponds to an actual HF Model ---
    try:
        hf_api = HfApi()
        logger.info(f"Attempting to fetch model info for: {normalized_model_id}")
        model_info = hf_api.model_info(normalized_model_id)
        logger.info(f"Successfully fetched model info for: {normalized_model_id}")
    except RepositoryNotFoundError:
        # Sanitize for safe HTML display in template
        sanitized_for_display = html.escape(normalized_model_id)
        error_message = f"Error: The provided ID \"{sanitized_for_display}\" could not be found on Hugging Face or does not correspond to a model repository."
        logger.warning(f"Repository not found for ID: {normalized_model_id}")
        return templates.TemplateResponse(
            "error.html", {"request": request, "error": error_message, "sbom_count": sbom_count, "model_id": sanitized_for_display}
        )
    except Exception as api_err:  # Catch other potential API errors
        sanitized_for_display = html.escape(normalized_model_id)
        error_message = f"Error verifying model ID with Hugging Face API: {str(api_err)}"
        logger.error(f"HF API error for {normalized_model_id}: {str(api_err)}")
        return templates.TemplateResponse(
            "error.html", {"request": request, "error": error_message, "sbom_count": sbom_count, "model_id": sanitized_for_display}
        )
    # --- End Model Existence Check ---

    
    # --- Main Generation Logic --- 
    try:
        # Try different import paths for AIBOMGenerator
        generator = None
        try:
            from src.aibom_generator.generator import AIBOMGenerator
            generator = AIBOMGenerator()
        except ImportError:
            try:
                from aibom_generator.generator import AIBOMGenerator
                generator = AIBOMGenerator()
            except ImportError:
                try:
                    from generator import AIBOMGenerator
                    generator = AIBOMGenerator()
                except ImportError:
                    logger.error("Could not import AIBOMGenerator from any known location")
                    raise ImportError("Could not import AIBOMGenerator from any known location")

        # Generate AIBOM (pass SANITIZED ID)
        aibom = generator.generate_aibom(
            model_id=sanitized_model_id, # Use sanitized ID
            include_inference=include_inference,
            use_best_practices=use_best_practices
        )
        enhancement_report = generator.get_enhancement_report()

        # Save AIBOM to file, use industry term ai_sbom in file name
        # Save AIBOM to file using normalized ID
        filename = f"{normalized_model_id.replace('/', '_')}_ai_sbom.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(aibom, f, indent=2)

        # --- Log Generation Event ---
        log_sbom_generation(sanitized_model_id) # Use sanitized ID
        sbom_count = get_sbom_count() # Refresh count after logging
        # --- End Log ---

        download_url = f"/output/{filename}"

        # Create download and UI interaction scripts
        download_script = f"""
        <script>
            function downloadJSON() {{
                const a = document.createElement('a');
                a.href = '{download_url}';
                a.download = '{filename}';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }}

            function switchTab(tabId) {{
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => {{
                    tab.classList.remove('active');
                }});

                // Deactivate all tab buttons
                document.querySelectorAll('.aibom-tab').forEach(button => {{
                    button.classList.remove('active');
                }});

                // Show the selected tab
                document.getElementById(tabId).classList.add('active');

                // Activate the clicked button
                event.currentTarget.classList.add('active');
            }}

            function toggleCollapsible(element) {{
                element.classList.toggle('active');
                var content = element.nextElementSibling;
                if (content.style.maxHeight) {{
                    content.style.maxHeight = null;
                    content.classList.remove('active');
                }} else {{
                    content.style.maxHeight = content.scrollHeight + "px";
                    content.classList.add('active');
                }}
            }}
        </script>
        """

        # Get completeness score or create a comprehensive one if not available
        # Use sanitized_model_id
        completeness_score = None
        if hasattr(generator, 'get_completeness_score'):
            try:
                completeness_score = generator.get_completeness_score(sanitized_model_id)
                logger.info("Successfully retrieved completeness_score from generator")
            except Exception as e:
                logger.error(f"Completeness score error from generator: {str(e)}")


        # If completeness_score is None or doesn't have field_checklist, use comprehensive one
        if completeness_score is None or not isinstance(completeness_score, dict) or 'field_checklist' not in completeness_score:
            logger.info("Using comprehensive completeness_score with field_checklist")
            completeness_score = create_comprehensive_completeness_score(aibom)

        # Ensure enhancement_report has the right structure
        if enhancement_report is None:
            enhancement_report = {
                "ai_enhanced": False,
                "ai_model": None,
                "original_score": {"total_score": 0, "completeness_score": 0},
                "final_score": {"total_score": 0, "completeness_score": 0},
                "improvement": 0
            }
        else:
            # Ensure original_score has completeness_score
            if "original_score" not in enhancement_report or enhancement_report["original_score"] is None:
                enhancement_report["original_score"] = {"total_score": 0, "completeness_score": 0}
            elif "completeness_score" not in enhancement_report["original_score"]:
                enhancement_report["original_score"]["completeness_score"] = enhancement_report["original_score"].get("total_score", 0)

            # Ensure final_score has completeness_score
            if "final_score" not in enhancement_report or enhancement_report["final_score"] is None:
                enhancement_report["final_score"] = {"total_score": 0, "completeness_score": 0}
            elif "completeness_score" not in enhancement_report["final_score"]:
                enhancement_report["final_score"]["completeness_score"] = enhancement_report["final_score"].get("total_score", 0)

        # Add display names and tooltips for score sections
        display_names = {
            "required_fields": "Required Fields",
            "metadata": "Metadata",
            "component_basic": "Component Basic Info",
            "component_model_card": "Model Card",
            "external_references": "External References"
        }

        tooltips = {
            "required_fields": "Basic required fields for a valid AIBOM",
            "metadata": "Information about the AIBOM itself",
            "component_basic": "Basic information about the AI model component",
            "component_model_card": "Detailed model card information",
            "external_references": "Links to external resources"
        }

        weights = {
            "required_fields": 20,
            "metadata": 20,
            "component_basic": 20,
            "component_model_card": 30,
            "external_references": 10
        }

        # DEBUG: Check for undefined values before template rendering
        print("DEBUG: Checking completeness_score for undefined values:")
        if completeness_score and 'section_scores' in completeness_score:
            for key, value in completeness_score['section_scores'].items():
                print(f"  {key}: {value} (type: {type(value)})")
        else:
            print("  No section_scores found in completeness_score")
        
        # DEBUG: Template data check
        print("DEBUG: Template data check:")
        if completeness_score:
            print(f"  completeness_score keys: {list(completeness_score.keys())}")
            if 'category_details' in completeness_score:
                print(f"  category_details exists: {list(completeness_score['category_details'].keys())}")
                # Use registry-driven categories when available
                if REGISTRY_AVAILABLE:
                    categories = set(classification["category"] for classification in FIELD_CLASSIFICATION.values())
                else:
                    categories = ['required_fields', 'metadata', 'component_basic', 'component_model_card', 'external_references']
                    
                for category in categories:
                    if category in completeness_score['category_details']:
                        details = completeness_score['category_details'][category]
                        print(f"    {category}: present={details.get('present_fields')}, total={details.get('total_fields')}, percentage={details.get('percentage')}")
                    else:
                        print(f"    {category}: MISSING from category_details")
            else:
                print("  category_details: NOT FOUND in completeness_score!")
        else:
            print("  completeness_score: IS NONE!")
        
        # Render the template with all necessary data, with sanitized model ID for display
        sanitized_for_display = html.escape(normalized_model_id)
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "model_id": sanitized_for_display,
                "aibom": aibom,
                "enhancement_report": enhancement_report,
                "completeness_score": completeness_score,
                "download_url": download_url,
                "download_script": download_script,
                "display_names": display_names,
                "tooltips": tooltips,
                "weights": weights,
                "sbom_count": sbom_count,
                "display_names": display_names,
                "tooltips": tooltips,
                "weights": weights
            }
        )
    # --- Main Exception Handling ---
    except Exception as e:
        logger.error(f"Error generating AI SBOM: {str(e)}")
        sbom_count = get_sbom_count()  # Refresh count just in case
        # Sanitize model ID for safe HTML display
        sanitized_for_display = html.escape(normalized_model_id)
        return templates.TemplateResponse(
            "error.html", {"request": request, "error": str(e), "sbom_count": sbom_count, "model_id": sanitized_for_display}
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a generated AIBOM file.

    This endpoint serves the generated AIBOM JSON files for download.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/json",
        filename=filename
    )

# Request model for JSON API
class GenerateRequest(BaseModel):
    model_id: str
    include_inference: bool = True
    use_best_practices: bool = True
    hf_token: Optional[str] = None

@app.post("/api/generate")
async def api_generate_aibom(request: GenerateRequest):
    """
    Generate an AI SBOM for a specified Hugging Face model.

    This endpoint accepts JSON input and returns JSON output.
    """
    try:
        # SECURITY: Validate raw input FIRST, before any sanitization
        # This prevents bypass scenarios where sanitized input might slip through
        if not is_valid_hf_input(request.model_id):
            sanitized_for_display = html.escape(request.model_id)
            logger.warning(f"Invalid model input format received: {sanitized_for_display}")
            raise HTTPException(status_code=400, detail="Invalid model ID format")

        # Only sanitize AFTER validation passes (for safe display/logging)
        sanitized_model_id = html.escape(request.model_id)
        normalized_model_id = _normalise_model_id(sanitized_model_id)
        
        # Verify model exists
        try:
            hf_api = HfApi()
            model_info = hf_api.model_info(normalized_model_id)
        except RepositoryNotFoundError:
            raise HTTPException(status_code=404, detail=f"Model {normalized_model_id} not found on Hugging Face")
        except Exception as api_err:
            raise HTTPException(status_code=500, detail=f"Error verifying model: {str(api_err)}")
        
        # Generate AIBOM
        try:
            # Try different import paths for AIBOMGenerator
            generator = None
            try:
                from src.aibom_generator.generator import AIBOMGenerator
                generator = AIBOMGenerator()
            except ImportError:
                try:
                    from aibom_generator.generator import AIBOMGenerator
                    generator = AIBOMGenerator()
                except ImportError:
                    try:
                        from generator import AIBOMGenerator
                        generator = AIBOMGenerator()
                    except ImportError:
                        raise HTTPException(status_code=500, detail="Could not import AIBOMGenerator")
        
            aibom = generator.generate_aibom(
                model_id=sanitized_model_id,
                include_inference=request.include_inference,
                use_best_practices=request.use_best_practices
            )
            enhancement_report = generator.get_enhancement_report()
            
            # Save AIBOM to file
            filename = f"{normalized_model_id.replace('/', '_')}_ai_sbom.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w") as f:
                json.dump(aibom, f, indent=2)
            
            # Log generation
            log_sbom_generation(sanitized_model_id)
            
            # Return JSON response
            return {
                "aibom": aibom,
                "model_id": normalized_model_id,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "request_id": str(uuid.uuid4()),
                "download_url": f"/output/{filename}"
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating AI SBOM: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI SBOM: {str(e)}")

@app.post("/api/generate-with-report")
async def api_generate_with_report(request: GenerateRequest):
    """
    Generate an AI SBOM with a completeness report.
    This endpoint accepts JSON input and returns JSON output with completeness score.
    """
    try:
        # SECURITY: Validate raw input FIRST, before any sanitization
        # This prevents bypass scenarios where sanitized input might slip through
        if not is_valid_hf_input(request.model_id):
            sanitized_for_display = html.escape(request.model_id)
            logger.warning(f"Invalid model input format received: {sanitized_for_display}")
            raise HTTPException(status_code=400, detail="Invalid model ID format")

        # Only sanitize AFTER validation passes (for safe display/logging)
        sanitized_model_id = html.escape(request.model_id)
        normalized_model_id = _normalise_model_id(sanitized_model_id)
        
        # Verify model exists
        try:
            hf_api = HfApi()
            model_info = hf_api.model_info(normalized_model_id)
        except RepositoryNotFoundError:
            raise HTTPException(status_code=404, detail=f"Model {normalized_model_id} not found on Hugging Face")
        except Exception as api_err:
            raise HTTPException(status_code=500, detail=f"Error verifying model: {str(api_err)}")
        
        # Generate AIBOM
        try:
            # Try different import paths for AIBOMGenerator
            generator = None
            try:
                from src.aibom_generator.generator import AIBOMGenerator
                generator = AIBOMGenerator()
            except ImportError:
                try:
                    from aibom_generator.generator import AIBOMGenerator
                    generator = AIBOMGenerator()
                except ImportError:
                    try:
                        from generator import AIBOMGenerator
                        generator = AIBOMGenerator()
                    except ImportError:
                        raise HTTPException(status_code=500, detail="Could not import AIBOMGenerator")
        
            aibom = generator.generate_aibom(
                model_id=sanitized_model_id,
                include_inference=request.include_inference,
                use_best_practices=request.use_best_practices
            )
            
            # Calculate completeness score
            try:
                completeness_score = calculate_completeness_score(aibom, validate=True, use_best_practices=True)
            except Exception as e:
                logger.error(f"Failed completeness scoring for {normalized_model_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error calculating score: {str(e)}")
            
            # Round only section_scores that aren't already rounded
            for section, score in completeness_score["section_scores"].items():
                if isinstance(score, float) and not score.is_integer():
                    completeness_score["section_scores"][section] = round(score, 1)
            
            # Convert field_checklist to machine-parseable format
            if "field_checklist" in completeness_score:
                machine_parseable_checklist = {}
                for field, value in completeness_score["field_checklist"].items():
                    # Extract presence (✔/✘) and importance (★★★/★★/★)
                    present = "present" if "✔" in value else "missing"
                    
                    # Use field_tiers for importance since it's already machine-parseable
                    importance = completeness_score["field_tiers"].get(field, "unknown")
                    
                    # Create structured entry
                    machine_parseable_checklist[field] = {
                        "status": present,
                        "importance": importance
                    }
                
                # Replace the original field_checklist with the machine-parseable version
                completeness_score["field_checklist"] = machine_parseable_checklist
            
            # Remove field_tiers to avoid duplication (now incorporated in field_checklist)
            completeness_score.pop("field_tiers", None)
            
            # Save AIBOM to file
            filename = f"{normalized_model_id.replace('/', '_')}_ai_sbom.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w") as f:
                json.dump(aibom, f, indent=2)
            
            # Log generation
            log_sbom_generation(sanitized_model_id)
            
            # Return JSON response with improved completeness score
            return {
                "aibom": aibom,
                "model_id": normalized_model_id,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "request_id": str(uuid.uuid4()),
                "download_url": f"/output/{filename}",
                "completeness_score": completeness_score
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating AI SBOM: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI SBOM: {str(e)}")


@app.get("/api/models/{model_id:path}/score")
async def get_model_score(
    model_id: str,
    hf_token: Optional[str] = None,
    use_best_practices: bool = True
):
    """
    Get the completeness score for a model without generating a full AIBOM.
    """
    try:
        # SECURITY: Validate raw input FIRST, before any sanitization
        # This prevents bypass scenarios where sanitized input might slip through
        if not is_valid_hf_input(model_id):
            sanitized_for_display = html.escape(model_id)
            logger.warning(f"Invalid model input format received: {sanitized_for_display}")
            raise HTTPException(status_code=400, detail="Invalid model ID format")

        # Only sanitize AFTER validation passes (for safe display/logging)
        sanitized_model_id = html.escape(model_id)
        normalized_model_id = _normalise_model_id(sanitized_model_id)
        
        # Verify model exists
        try:
            hf_api = HfApi(token=hf_token)
            model_info = hf_api.model_info(normalized_model_id)
        except RepositoryNotFoundError:
            raise HTTPException(status_code=404, detail=f"Model {normalized_model_id} not found on Hugging Face")
        except Exception as api_err:
            raise HTTPException(status_code=500, detail=f"Error verifying model: {str(api_err)}")
        
        # Generate minimal AIBOM for scoring
        try:
            # Try different import paths for AIBOMGenerator
            generator = None
            try:
                from src.aibom_generator.generator import AIBOMGenerator
                generator = AIBOMGenerator(hf_token=hf_token)
            except ImportError:
                try:
                    from aibom_generator.generator import AIBOMGenerator
                    generator = AIBOMGenerator(hf_token=hf_token)
                except ImportError:
                    try:
                        from generator import AIBOMGenerator
                        generator = AIBOMGenerator(hf_token=hf_token)
                    except ImportError:
                        raise HTTPException(status_code=500, detail="Could not import AIBOMGenerator")
            
            # Generate minimal AIBOM
            aibom = generator.generate_aibom(
                model_id=sanitized_model_id,
                include_inference=False,  # No need for inference for just scoring
                use_best_practices=use_best_practices
            )
            
            # Calculate score
            score = calculate_completeness_score(aibom, validate=True, use_best_practices=use_best_practices)

            # Log SBOM generation for counting purposes
            log_sbom_generation(normalized_model_id)
            
            # Round section scores for better readability
            for section, value in score["section_scores"].items():
                if isinstance(value, float) and not value.is_integer():
                    score["section_scores"][section] = round(float(value), 1) if value is not None and value != "Undefined" else 0.0
            
            # Return score information
            return {
                "total_score": score["total_score"],
                "section_scores": score["section_scores"],
                "max_scores": score["max_scores"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating model score: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# Batch request model
class BatchRequest(BaseModel):
    model_ids: List[str]
    include_inference: bool = True
    use_best_practices: bool = True
    hf_token: Optional[str] = None

# In-memory storage for batch jobs
batch_jobs = {}

@app.post("/api/batch")
async def batch_generate(request: BatchRequest):
    """
    Start a batch job to generate AIBOMs for multiple models.
    """
    try:
        # SECURITY: Validate raw input FIRST, before any sanitization
        # This prevents bypass scenarios where sanitized input might slip through
        valid_model_ids = []
        for model_id in request.model_ids:
            if is_valid_hf_input(model_id):
                # Only sanitize AFTER validation passes
                sanitized_id = html.escape(model_id)
                valid_model_ids.append(sanitized_id)
            else:
                sanitized_for_display = html.escape(model_id)
                logger.warning(f"Skipping invalid model ID: {sanitized_for_display}")
        
        if not valid_model_ids:
            raise HTTPException(status_code=400, detail="No valid model IDs provided")
        
        # Create job ID
        job_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Store job information
        batch_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "model_ids": valid_model_ids,
            "created_at": created_at.isoformat() + "Z",
            "completed": 0,
            "total": len(valid_model_ids),
            "results": {}
        }
        
        # Would be best to start a background task here but for now marking it as "processing"
        batch_jobs[job_id]["status"] = "processing"
        
        return {
            "job_id": job_id,
            "status": "queued",
            "model_ids": valid_model_ids,
            "created_at": created_at.isoformat() + "Z"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating batch job: {str(e)}")

@app.get("/api/batch/{job_id}")
async def get_batch_status(job_id: str):
    """
    Check the status of a batch job.
    """
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail=f"Batch job {job_id} not found")
    
    return batch_jobs[job_id]


# If running directly (for local testing)
if __name__ == "__main__":
    import uvicorn
    # Ensure HF_TOKEN is set for local testing if needed
    if not HF_TOKEN:
        print("Warning: HF_TOKEN environment variable not set. SBOM count will show N/A and logging will be skipped.")
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.get("/api/registry/status")
async def get_registry_status():
    """Get current registry configuration status for debugging."""
    if REGISTRY_AVAILABLE:
        categories = {}
        for field_name, classification in FIELD_CLASSIFICATION.items():
            category = classification["category"]
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
        return {
            "registry_available": True,
            "total_fields": len(FIELD_CLASSIFICATION),
            "categories": list(categories.keys()),
            "field_count_by_category": categories,
            "registry_manager_loaded": REGISTRY_MANAGER is not None
        }
    else:
        return {
            "registry_available": False,
            "fallback_mode": True,
            "message": "Using hardcoded field definitions",
            "total_fields": 6  # Hardcoded fallback count
        }