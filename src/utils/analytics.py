import os
import logging
from datetime import datetime
import queue
import threading
import atexit
from datasets import Dataset, load_dataset, concatenate_datasets
from ..config import HF_REPO, HF_TOKEN

logger = logging.getLogger(__name__)

_log_queue = queue.Queue()
_worker_thread = None
_worker_lock = threading.Lock()


def _push_batch(batch):
    if not batch:
        return
    try:
        log_data = {
            "timestamp": [item["timestamp"] for item in batch],
            "event": [item["event"] for item in batch],
            "model_id": [item["model_id"] for item in batch],
        }
        ds_new_log = Dataset.from_dict(log_data)

        try:
            existing_ds = load_dataset(
                HF_REPO, token=HF_TOKEN, split="train", trust_remote_code=True
            )
            if len(existing_ds) > 0:
                ds_to_push = concatenate_datasets([existing_ds, ds_new_log])
            else:
                ds_to_push = ds_new_log
        except Exception as load_err:
            logger.info(f"Could not load existing dataset: {load_err}. Creating new.")
            ds_to_push = ds_new_log

        ds_to_push.push_to_hub(HF_REPO, token=HF_TOKEN, private=True)
        logger.info(
            f"Successfully logged SBOM generation for {len(batch)} models in a batch"
        )
    except Exception as e:
        logger.error(f"Background analytics batch push failed: {e}")


def _analytics_worker():
    while True:
        try:
            # Block until at least one item is available
            first_item = _log_queue.get(timeout=10)
            batch = [first_item]
            _log_queue.task_done()

            # Try to grab more items if they are available immediately
            while True:
                try:
                    item = _log_queue.get_nowait()
                    batch.append(item)
                    _log_queue.task_done()
                except queue.Empty:
                    break

            # Push the batch
            _push_batch(batch)

        except queue.Empty:
            # Loop again, wait for next item
            continue
        except Exception as e:
            logger.error(f"Analytics worker encountered an error: {e}")


def _flush_logs():
    batch = []
    while not _log_queue.empty():
        try:
            batch.append(_log_queue.get_nowait())
            _log_queue.task_done()
        except queue.Empty:
            break
    if batch:
        logger.info(f"Flushing {len(batch)} analytics logs on exit...")
        _push_batch(batch)


atexit.register(_flush_logs)


def log_sbom_generation(model_id: str):
    """Logs a successful SBOM generation event to the Hugging Face dataset."""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set. Skipping SBOM generation logging.")
        return

    try:
        global _worker_thread

        # Start worker thread if not running
        if _worker_thread is None or not _worker_thread.is_alive():
            with _worker_lock:
                if _worker_thread is None or not _worker_thread.is_alive():
                    _worker_thread = threading.Thread(
                        target=_analytics_worker, daemon=True
                    )
                    _worker_thread.start()

        # Enqueue the log item
        log_item = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "generated",
            "model_id": model_id,
        }
        _log_queue.put(log_item)

    except Exception as e:
        logger.error(f"Failed to initiate analytics logging: {e}")


def get_sbom_count() -> str:
    """Retrieves the total count of generated SBOMs."""
    if not HF_TOKEN:
        return "N/A"
    try:
        ds = load_dataset(
            HF_REPO, token=HF_TOKEN, split="train", trust_remote_code=True
        )
        # We can also add the un-pushed queue size to make it more accurate
        return f"{len(ds) + _log_queue.qsize():,}"
    except Exception as e:
        logger.error(f"Failed to retrieve SBOM count: {e}")
        return "N/A"
