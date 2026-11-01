import os
import re
import json
import logging
import html
from urllib.parse import urlparse
from typing import Optional
import aiofiles

from fastapi import APIRouter, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from ..models.service import AIBOMService
from ..models.scoring import calculate_completeness_score
from ..utils.analytics import log_sbom_generation, get_sbom_count
from ..utils.formatter import export_aibom
from ..config import TEMPLATES_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Helpers ---
HF_SEGMENT_REGEX = re.compile(r"[a-zA-Z0-9\.\-\_]+")


def _is_valid_hf_segment(segment: str) -> bool:
    # fullmatch (not `$`, which also matches before a trailing newline); "." and ".."
    # are path-navigation tokens, never valid Hugging Face owner or repo names.
    return segment not in (".", "..") and HF_SEGMENT_REGEX.fullmatch(segment) is not None


def is_valid_hf_input(input_str: str) -> bool:
    """Validate the raw, unsanitized user input (must run before html.escape)."""
    if not input_str or len(input_str) > 200:
        return False
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in input_str):
        return False
    if input_str.startswith(("http://", "https://")):
        try:
            parsed = urlparse(input_str)
            if parsed.netloc == "huggingface.co":
                parts = parsed.path.strip("/").split("/")
                if len(parts) >= 2 and _is_valid_hf_segment(parts[0]) and _is_valid_hf_segment(parts[1]):
                    return True
            return False
        except Exception:
            return False
    owner, sep, repo = input_str.partition("/")
    return bool(sep) and _is_valid_hf_segment(owner) and _is_valid_hf_segment(repo)

# --- Routes ---

@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "request": request, 
        "sbom_count": get_sbom_count()
    })

@router.get("/status")
async def get_status():
    return {"status": "operational", "version": "1.0.0", "generator_version": "2.0.0"}

@router.post("/generate", response_class=HTMLResponse)
async def generate_form(
    request: Request,
    model_id: str = Form(...),
    include_inference: bool = Form(False),
    use_best_practices: bool = Form(True)
):
    # Security: Validate BEFORE sanitizing to prevent bypass attacks
    # (e.g., <script>org/model</script> → &lt;script&gt;org/model&lt;/script&gt; could slip through)
    if not is_valid_hf_input(model_id):
        return templates.TemplateResponse(request, "error.html", {
            "request": request,
            "error": "Invalid model ID format.",
            "sbom_count": get_sbom_count(),
            "model_id": html.escape(model_id)
        })

    # Sanitize after validation for safe display/storage
    sanitized_model_id = html.escape(model_id)
    
    # Use helper from Service to normalize
    normalized_id = AIBOMService._normalise_model_id(sanitized_model_id)

    # Check existence (non-blocking)
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: HfApi().model_info(normalized_id))
    except RepositoryNotFoundError:
        return templates.TemplateResponse(request, "error.html", {
            "request": request,
            "error": f"Model {normalized_id} not found on Hugging Face.",
            "sbom_count": get_sbom_count(),
            "model_id": normalized_id
        })
    except Exception as e:
        return templates.TemplateResponse(request, "error.html", {
            "request": request,
            "error": f"Error verifying model: {e}",
            "sbom_count": get_sbom_count(),
            "model_id": normalized_id
        })

    # Generate (non-blocking)
    try:
        def _generate_task():
            service = AIBOMService(use_best_practices=use_best_practices)
            aibom = service.generate_aibom(sanitized_model_id, include_inference=include_inference)
            report = service.get_enhancement_report()
            return service, aibom, report

        service, aibom, report = await loop.run_in_executor(None, _generate_task)
        
        # Save file (non-blocking I/O)
        filename = f"{normalized_id.replace('/', '_')}_ai_sbom_1_6.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        filepath_1_7 = os.path.join(OUTPUT_DIR, f"{normalized_id.replace('/', '_')}_ai_sbom_1_7.json")
        
        def _format_task():
            # Generate Formatted JSON strings
            json_1_6 = export_aibom(aibom, bom_type="cyclonedx", spec_version="1.6")
            json_1_7 = export_aibom(aibom, bom_type="cyclonedx", spec_version="1.7")
            return json_1_6, json_1_7
            
        json_1_6, json_1_7 = await loop.run_in_executor(None, _format_task)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
            await f.write(json_1_6)
        async with aiofiles.open(filepath_1_7, "w", encoding="utf-8") as f:
            await f.write(json_1_7)
        log_sbom_generation(sanitized_model_id)
        
        # Extract score
        completeness_score = None
        if report and "final_score" in report:
            completeness_score = report["final_score"]
        
        # Fallback score if needed
        if not completeness_score:
            completeness_score = calculate_completeness_score(aibom)

        # Prepare context for template
        context = {
            "request": request,
            "filename": filename,
            "download_url": f"/output/{filename}",
            "aibom": aibom,
            "aibom_cdx_json_1_6": json_1_6,
            "aibom_cdx_json_1_7": json_1_7,
            "components_json": json.dumps(aibom.get("components", []), indent=2),
            "model_id": normalized_id,
            "sbom_count": get_sbom_count(),
            "completeness_score": completeness_score,
            "enhancement_report": report or {},
            # Pass legacy variables for template compatibility if needed
            "result_file": f"/output/{filename}" 
        }
        
        return templates.TemplateResponse(request, "result.html", context)

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return templates.TemplateResponse(request, "error.html", {
            "request": request,
            "error": f"Internal generation error: {e}",
            "sbom_count": get_sbom_count(),
            "model_id": normalized_id
        })
