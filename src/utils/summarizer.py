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
    #_model_name = "sshleifer/distilbart-cnn-12-6"
    _model_name = "google/flan-t5-small"

    @classmethod
    def _load_model(cls):
        """Lazy load the model and tokenizer directly"""
        if cls._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import transformers
                logger.info(f"⏳ Loading summarization model ({cls._model_name})...")
                
                # Temporarily suppress transformers warnings (like tie_word_embeddings warning)
                old_verbosity = transformers.logging.get_verbosity()
                transformers.logging.set_verbosity_error()
                
                cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
                cls._model = AutoModelForSeq2SeqLM.from_pretrained(cls._model_name)
                
                transformers.logging.set_verbosity(old_verbosity)
                
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
            import re
            from bs4 import BeautifulSoup
            
            # Remove HTML tags and CSS
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["style", "script"]):
                tag.decompose()
            text = soup.get_text(separator=' ')
            
            # Remove markdown tables
            text = re.sub(r'\|.*?\|', '', text)
            text = re.sub(r'(?m)^\|.*\|$', '', text)
            
            # Remove excessive whitespace and rudimentary markdown tokens
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text) 
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Truncate input broadly, 1000 chars is sufficient for summary and avoids trailing licenses/params
            truncated_input = text[:1000]
            
            # Prepare inputs
            # T5 model task prefix
            if "t5" in cls._model_name.lower() or "flan" in cls._model_name.lower():
                truncated_input = f"Summarize the following AI model description:\n\n{truncated_input}"

            inputs = cls._tokenizer(truncated_input, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate
            generate_kwargs = {
                "max_length": 250,
                "do_sample": False,
                "num_beams": 4,
                "early_stopping": True,
                "repetition_penalty": 1.5
            }
            
            # Add min_length for non-T5 models, as T5 can struggle with forced min_lengths
            if "t5" not in cls._model_name.lower() and "flan" not in cls._model_name.lower():
                generate_kwargs["min_length"] = 30

            summary_ids = cls._model.generate(
                inputs["input_ids"],
                **generate_kwargs
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
