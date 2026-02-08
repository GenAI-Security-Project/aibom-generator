import os
import logging
from datetime import datetime
from datasets import Dataset, load_dataset, concatenate_datasets
from ..config import HF_REPO, HF_TOKEN

logger = logging.getLogger(__name__)

def log_sbom_generation(model_id: str):
    """Logs a successful SBOM generation event to the Hugging Face dataset."""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set. Skipping SBOM generation logging.")
        return

    try:
        if not HF_TOKEN:
            return

        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Define the synchronous task
        def _push_log():
            try:
                normalized_model_id = model_id 
                log_data = {
                    "timestamp": [datetime.utcnow().isoformat()],
                    "event": ["generated"],
                    "model_id": [normalized_model_id]
                }
                ds_new_log = Dataset.from_dict(log_data)

                # Optimisation: Try to append if possible, but datasets library is heavy.
                # Just catch errors to ensure main thread never crashes.
                try:
                    existing_ds = load_dataset(HF_REPO, token=HF_TOKEN, split='train', trust_remote_code=True)
                    if len(existing_ds) > 0:
                         ds_to_push = concatenate_datasets([existing_ds, ds_new_log])
                    else:
                         ds_to_push = ds_new_log
                except Exception as load_err:
                     logger.info(f"Could not load existing dataset: {load_err}. Creating new.")
                     ds_to_push = ds_new_log

                ds_to_push.push_to_hub(HF_REPO, token=HF_TOKEN, private=True)
                logger.info(f"Successfully logged SBOM generation for {model_id}")
            except Exception as e:
                logger.error(f"Background analytics failed: {e}")

        # Fire and forget in a separate thread
        # Use existing event loop if available, else fire in thread
        loop = None
        try:
             loop = asyncio.get_running_loop()
        except RuntimeError:
             pass
        
        if loop and loop.is_running():
            loop.run_in_executor(None, _push_log)
        else:
             # Fallback for sync contexts (like CLI)
             ThreadPoolExecutor(max_workers=1).submit(_push_log)

    except Exception as e:
        logger.error(f"Failed to initiate analytics logging: {e}")

def get_sbom_count() -> str:
    """Retrieves the total count of generated SBOMs."""
    if not HF_TOKEN:
        return "N/A"
    try:
        ds = load_dataset(HF_REPO, token=HF_TOKEN, split='train', trust_remote_code=True)
        return f"{len(ds):,}"
    except Exception as e:
        logger.error(f"Failed to retrieve SBOM count: {e}")
        return "N/A"
