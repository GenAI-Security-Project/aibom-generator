import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .config import OUTPUT_DIR, MAX_AGE_DAYS, MAX_FILES, CLEANUP_INTERVAL
from .controllers.web_controller import router as web_router
from .utils import RateLimitMiddleware, ConcurrencyLimitMiddleware, RequestSizeLimitMiddleware, perform_cleanup

# Ensure registry is initialized
from .models import get_field_registry_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aibom_generator")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AI SBOM Generator WebApp")
    try:
        get_field_registry_manager() # Ensure registry is loaded
        logger.info("Registry loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")
        
    # Initial cleanup
    try:
        perform_cleanup(OUTPUT_DIR, MAX_AGE_DAYS, MAX_FILES)
    except Exception as e:
        logger.warning(f"Initial cleanup failed: {e}")
        
    yield
    # Shutdown (if needed)

app = FastAPI(title="AI SBOM Generator", lifespan=lifespan)

# --- Middleware ---
app.add_middleware(
    RateLimitMiddleware,
    rate_limit_per_minute=10,
    rate_limit_window=60,
    protected_routes=["/generate"]
)
app.add_middleware(
    ConcurrencyLimitMiddleware,
    max_concurrent_requests=5,
    timeout=5.0,
    protected_routes=["/generate"]
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_content_length=1024*1024  # 1MB
)

# --- Cleanup Middleware ---
request_counter = 0

@app.middleware("http")
async def cleanup_middleware(request: Request, call_next):
    global request_counter
    request_counter += 1
    if request_counter % CLEANUP_INTERVAL == 0:
        try:
            removed = perform_cleanup(OUTPUT_DIR, MAX_AGE_DAYS, MAX_FILES)
            logger.info(f"Scheduled cleanup removed {removed} files")
        except Exception as e:
            logger.error(f"Error during scheduled cleanup: {e}")
            
    response = await call_next(request)
    return response

# --- Static Files ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
# Mount static files (CSS/JS)
os.makedirs("src/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# --- Routes ---
app.include_router(web_router)

if __name__ == "__main__":
    import uvicorn
    # Print clear access URL to avoid 0.0.0.0 confusion
    print("🚀 Application ready! Access it at: http://localhost:8000")
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
