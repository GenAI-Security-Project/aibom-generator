import os
from pathlib import Path

# Base Directory Setup
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = os.getenv("AIBOM_OUTPUT_DIR") or "/tmp/aibom_output"
# Ensure absolute path for security
if not os.path.isabs(OUTPUT_DIR):
    OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)

TEMPLATES_DIR = BASE_DIR / "templates"

# Cleanup Configuration
MAX_AGE_DAYS = 7
MAX_FILES = 1000
CLEANUP_INTERVAL = 100

# Hugging Face Setup
HF_REPO = "owasp-genai-security-project/aisbom-usage-log"
HF_TOKEN = os.getenv("HF_TOKEN")
RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY")
