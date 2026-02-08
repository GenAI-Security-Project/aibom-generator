import os
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__ )

# Get the secret key from environment variable
RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY")

def verify_recaptcha(response_token: Optional[str]) -> bool:
    # LOGGING: Log the token start
    logger.info(f"Starting reCAPTCHA verification with token: {response_token[:10]}..." if response_token else "None")
    
    # Check if secret key is set
    secret_key = os.environ.get("RECAPTCHA_SECRET_KEY")
    if not secret_key:
        logger.warning("RECAPTCHA_SECRET_KEY not set, bypassing verification")
        return True
    else:
        # LOGGING: Log that secret key is set
        logger.info("RECAPTCHA_SECRET_KEY is set (not showing for security)")
        
    # If no token provided, verification fails
    if not response_token:
        logger.warning("No reCAPTCHA response token provided")
        return False
    
    try:
        # LOGGING: Log before making request
        logger.info("Sending verification request to Google reCAPTCHA API")
        verification_response = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={
                "secret": secret_key,
                "response": response_token
            }
         )
        
        result = verification_response.json()
        # LOGGING: Log the complete result from Google
        logger.info(f"reCAPTCHA verification result: {result}")
        
        if result.get("success"):
            logger.info("reCAPTCHA verification successful")
            return True
        else:
            # LOGGING: Log the specific error codes
            logger.warning(f"reCAPTCHA verification failed: {result.get('error-codes', [])}")
            return False
    except Exception as e:
        # LOGGING: Log any exceptions
        logger.error(f"Error verifying reCAPTCHA: {str(e)}")
        return False

