"""Per-file SHA-256 hashes for the files of a model repository (issue #138).

Every file needed to load and run the model at a pinned revision gets a content
hash: weights, tokenizer files, config and generation config, the chat template
wherever it lives, and custom code. Documentation files are out of scope.

Hash sources:
  * ``hf-lfs``   - the ``lfs.sha256`` the Hugging Face API reports for an LFS file,
                   which is the SHA-256 of the file content.
  * ``computed`` - for non-LFS files the API only gives the git blob SHA-1, not a
                   hash of the content, so the file is downloaded at the pinned
                   revision and hashed here.
  * ``unknown``  - no hash could be obtained (gated repo without a token, a failed
                   download, a file over the size limit). The file is still listed.
"""

import hashlib
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError

logger = logging.getLogger(__name__)

HASH_SOURCE_LFS = "hf-lfs"
HASH_SOURCE_COMPUTED = "computed"
HASH_SOURCE_UNKNOWN = "unknown"

# Non-LFS files larger than this are not downloaded for hashing. Files above the
# Hub's LFS threshold normally come with lfs.sha256, so this only guards against
# unusual repositories.
MAX_COMPUTED_FILE_BYTES = 50 * 1024 * 1024
# Upper bound on non-LFS downloads per repository.
MAX_COMPUTED_FILES = 200

_DOC_BASENAMES = {
    ".gitattributes", ".gitignore", "license", "licence", "notice", "copying",
    "use_policy", "responsible_use_guide",
}
_DOC_EXTENSIONS = {
    ".md", ".markdown", ".rst", ".pdf",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
}

Downloader = Callable[..., str]

_GATED_REASON = "gated repository: no access with the token used"


def is_documentation_file(path: str) -> bool:
    """Return True for files that are not needed to load or run the model."""
    base = os.path.basename(path).lower()
    stem, ext = os.path.splitext(base)
    if base in _DOC_BASENAMES or stem in _DOC_BASENAMES:
        return True
    return ext in _DOC_EXTENSIONS


def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_model_file_hashes(
    model_id: str,
    revision: Optional[str],
    hf_api: Optional[HfApi] = None,
    downloader: Optional[Downloader] = None,
    max_file_bytes: int = MAX_COMPUTED_FILE_BYTES,
    max_computed_files: int = MAX_COMPUTED_FILES,
) -> List[Dict[str, Any]]:
    """List the model's files at ``revision`` with a SHA-256 and its source.

    Returns a list of dicts with keys ``path``, ``sha256`` (None when unknown),
    ``source`` and, when known, ``size``; unknown entries also carry ``reason``.
    Returns an empty list when there is no pinned revision or the file listing
    can't be fetched: hashes taken from a moving branch are not reproducible.
    """
    if not revision:
        logger.info(f"No pinned revision for {model_id}; skipping model file hashes")
        return []

    api = hf_api or HfApi()
    try:
        info = api.model_info(model_id, revision=revision, files_metadata=True)
    except Exception as e:
        logger.warning(f"Model file listing failed for {model_id}@{revision}: {e}")
        return []

    download = downloader or hf_hub_download
    token = getattr(api, "token", None)
    computed = 0
    gated = False
    results: List[Dict[str, Any]] = []

    siblings = sorted(getattr(info, "siblings", None) or [], key=lambda s: getattr(s, "rfilename", ""))
    for sibling in siblings:
        path = getattr(sibling, "rfilename", None)
        if not path or is_documentation_file(path):
            continue

        size = getattr(sibling, "size", None)
        entry: Dict[str, Any] = {"path": path}
        if isinstance(size, int):
            entry["size"] = size

        lfs = getattr(sibling, "lfs", None)
        lfs_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        if lfs is not None:
            if _is_sha256(lfs_sha256):
                entry.update(sha256=lfs_sha256.lower(), source=HASH_SOURCE_LFS)
            else:
                entry.update(sha256=None, source=HASH_SOURCE_UNKNOWN,
                             reason="LFS file without a content SHA-256 in the API response")
            results.append(entry)
            continue

        if gated:
            entry.update(sha256=None, source=HASH_SOURCE_UNKNOWN, reason=_GATED_REASON)
        elif isinstance(size, int) and size > max_file_bytes:
            entry.update(sha256=None, source=HASH_SOURCE_UNKNOWN,
                         reason=f"non-LFS file larger than {max_file_bytes} bytes")
        elif computed >= max_computed_files:
            entry.update(sha256=None, source=HASH_SOURCE_UNKNOWN,
                         reason=f"more than {max_computed_files} non-LFS files")
        else:
            computed += 1
            try:
                local_path = download(repo_id=model_id, filename=path, revision=revision, token=token)
                entry.update(sha256=sha256_of_file(local_path), source=HASH_SOURCE_COMPUTED)
            except GatedRepoError:
                # No access with this token: every other non-LFS download would fail the same way.
                gated = True
                logger.warning(f"{model_id} is gated; non-LFS files can't be hashed without access")
                entry.update(sha256=None, source=HASH_SOURCE_UNKNOWN, reason=_GATED_REASON)
            except Exception as e:
                logger.warning(f"Could not download {path} from {model_id}@{revision}: {e}")
                entry.update(sha256=None, source=HASH_SOURCE_UNKNOWN, reason="download failed")
        results.append(entry)

    return results


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
