import logging
from typing import Optional

logger = logging.getLogger(__name__)

class LocalSummarizer:
    """
    Singleton-style wrapper for local LLM summarization to ensure lazy loading
    and efficient resource usage.
    """
    _tokenizer = None
    _model = None
    _model_name = "sshleifer/distilbart-cnn-12-6"

    @classmethod
    def _load_model(cls):
        """Lazy load the model and tokenizer directly"""
        if cls._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                logger.info(f"⏳ Loading summarization model ({cls._model_name})...")
                cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
                cls._model = AutoModelForSeq2SeqLM.from_pretrained(cls._model_name)
                logger.info("✅ Summarization model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load summarization model: {e}")
                cls._model = False # Mark as failed

    @classmethod
    def summarize(cls, text: str, max_output_chars: int = 256) -> Optional[str]:
        """
        Generate a summary of the provided text.
        """
        if not text or not text.strip():
            return None

        # Load model if not already loaded
        if cls._model is None:
            cls._load_model()
            
        if not cls._model or not cls._tokenizer:
            return None

        try:
            # Truncate input broadly
            truncated_input = text[:2500]
            
            # Prepare inputs
            # DistilBART doesn't need "summarize: " prefix
            inputs = cls._tokenizer(truncated_input, return_tensors="pt", max_length=1024, truncation=True)
            
            # Generate
            summary_ids = cls._model.generate(
                inputs["input_ids"],
                max_length=120,
                min_length=30,
                do_sample=False
            )
            
            summary = cls._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
            # Enforce character limit
            if len(summary) > max_output_chars:
                return summary[:max_output_chars-3] + "..."
            return summary
                
        except Exception as e:
            logger.warning(f"⚠️ Summarization failed: {e}")
            return None
            
        return None
