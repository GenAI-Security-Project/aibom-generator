import logging
import re
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class LocalSummarizer:
    """
    Singleton-style wrapper for local LLM summarization.
    Enhances extraction using robust heuristic rules and LLM generation with retry logic.
    """
    _tokenizer = None
    _model = None
    _model_name = "google/flan-t5-small"

    @classmethod
    def _load_model(cls):
        """Lazy load the model and tokenizer directly"""
        if cls._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import transformers
                logger.info(f"⏳ Loading summarization model ({cls._model_name})...")
                
                old_verbosity = transformers.logging.get_verbosity()
                transformers.logging.set_verbosity_error()
                
                cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
                cls._model = AutoModelForSeq2SeqLM.from_pretrained(cls._model_name)
                
                transformers.logging.set_verbosity(old_verbosity)
                logger.info("✅ Summarization model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load summarization model: {e}")
                cls._model = False # Mark as failed

    @staticmethod
    def _strip_yaml_frontmatter(text: str) -> str:
        """Strip the YAML frontmatter enclosed in ---"""
        return re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.MULTILINE | re.DOTALL)

    @staticmethod
    def _extract_candidates(text: str) -> List[str]:
        candidates = []
        
        # 1. Section Headers (support "1. Introduction")
        heading_matches = re.finditer(r'^#+\s*(?:\d+[\.\)]?\s*)?(Description|Model [dD]escription|Model Overview|Overview|Introduction|Summary|モデル概要|Model Details)[^\n]*\n(.*?)(?=\n#+\s|\Z)', text, flags=re.MULTILINE | re.DOTALL)
        for match in heading_matches:
            if match.group(2).strip():
                candidates.append(match.group(2).strip())
                
        # 2. Inline Labels
        inline_matches = re.finditer(r'(?:Description:|Overview:|### Description:)\s*(.*?)(?=\n\n|\Z)', text, flags=re.DOTALL | re.IGNORECASE)
        for match in inline_matches:
            if match.group(1).strip():
                candidates.append(match.group(1).strip())
                
        # 3. Auto-generated fine-tuned leading sentences
        tuned_matches = re.finditer(r'^(?:The .*model is a .*|This model is a fine-tuned version of.*|This is a fine-tuned.*)', text, flags=re.MULTILINE | re.IGNORECASE)
        for match in tuned_matches:
            candidates.append(match.group(0).strip())
            
        # 4. Fallback: First meaningful paragraph
        # Strip some HTML first just for the fallback rule
        html_stripped = re.sub(r'<[^>]+>', '', text)
        paragraphs = re.split(r'\n\s*\n', html_stripped)
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if p.startswith('#'):
                continue
            # Skip heavy markdown like links/images/badges and github alerts
            if p.startswith('[!') or p.startswith('<a href') or p.startswith('> [!'):
                continue
            # If a paragraph has many links (like a table of contents / link directory)
            if p.count('](') > 3 or p.count('http') > 3:
                continue
            if len(p) > 50:
                candidates.append(p)
                break
                
        return candidates

    @staticmethod
    def _score_candidate(text: str) -> float:
        score = 0.0
        text_lower = text.lower()
        
        # Length score (sweet spot between 100 and 500 chars)
        if 50 < len(text) < 1000:
            score += 10.0
            
        # Reward definitional patterns
        if "is a" in text_lower or "fine-tuned version of" in text_lower or "trained on" in text_lower or "designed for" in text_lower:
            score += 20.0
            
        # Penalize bad patterns
        if "leaderboard" in text_lower or "benchmark" in text_lower or "results" in text_lower:
            score -= 50.0
        if "install" in text_lower or "how to run" in text_lower or "pip install" in text_lower or "read our guide" in text_lower:
            score -= 30.0
            
        # Penalize table/code-heavy paragraphs and bullet points
        if text.count('|') > 5 or text.count('```') >= 1 or text.count('\n- ') > 2 or text.count('\n* ') > 2:
            score -= 50.0
            
        return score

    @staticmethod
    def _clean_text(text: str) -> str:
        # Remove HTML
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["style", "script"]):
                tag.decompose()
            text = soup.get_text(separator=' ')
        except Exception:
            pass
            
        # Remove markdown images
        text = re.sub(r'!\[.*?\]\([^)]+\)', '', text)
        # Convert links to just text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r'`[^`]*`', '', text)
        # Remove tables
        text = re.sub(r'\|.*?\|', '', text)
        text = re.sub(r'(?m)^[-:| ]+$', '', text) # table separators
        
        # Remove boilerplate line by line
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line_lower = line.lower()
            if 'generated automatically' in line_lower and 'model card' in line_lower:
                continue
            if 'completed by the model author' in line_lower:
                continue
            if 'model cards for model reporting' in line_lower:
                continue
            clean_lines.append(line)
        text = '\n'.join(clean_lines)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @classmethod
    def _generate(cls, prompt: str, max_output_chars: int) -> Optional[str]:
        if cls._model is None:
            cls._load_model()
        if not cls._model or not cls._tokenizer:
            return None
            
        try:
            inputs = cls._tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            generate_kwargs = {
                "max_length": 128,  # Increased by ~30% from 64
                "min_length": 15,  # Avoid single word outputs
                "do_sample": False,
                "num_beams": 4,
                "early_stopping": True,
                "repetition_penalty": 2.0
            }
            summary_ids = cls._model.generate(inputs["input_ids"], **generate_kwargs)
            summary = cls._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            summary = summary.strip()
            
            # Remove "Output:" prefix if present
            if summary.lower().startswith("output:"):
                summary = re.sub(r'^Output:\s*', '', summary, flags=re.IGNORECASE)
                
            if len(summary) > max_output_chars:
                return summary[:max_output_chars-3] + "..."
            return summary
        except Exception as e:
            logger.warning(f"⚠️ Generation failed: {e}")
            return None

    @staticmethod
    def _is_valid_summary(summary: str, model_id: str) -> bool:
        if not summary or len(summary) < 15:
            return False
            
        summary_lower = summary.lower()
        model_name = model_id.split('/')[-1].lower()
        
        if summary_lower == model_name or summary_lower == f"{model_name} model":
            return False
            
        # Check for markdown/html artifacts
        if '#' in summary or '<' in summary or '>' in summary or '*' in summary:
            return False
            
        # Check for instruction-like text
        if summary_lower.startswith("to install") or summary_lower.startswith("how to") or "pip install" in summary_lower:
            return False
            
        # Refuse literally copying bullet points (e.g. from table)
        if "- type:" in summary_lower or "number of parameters:" in summary_lower:
            return False
            
        return True

    @classmethod
    def summarize(cls, text: str, max_output_chars: int = 332, model_id: str = "") -> Optional[str]:
        """
        Robustly extract and summarize model description.
        """
        if not text or not text.strip():
            return None
            
        # 1. Strip YAML safely
        text_without_yaml = cls._strip_yaml_frontmatter(text)
        
        # 2. Extract multiple candidate description blocks
        candidates = cls._extract_candidates(text_without_yaml)
        
        if not candidates:
            # Fallback if candidates are absolutely empty
            candidates = [text_without_yaml[:1000]]
            
        # 3. Score candidates and pick best
        scored_candidates = [(c, cls._score_candidate(c)) for c in candidates]
        best_candidate = max(scored_candidates, key=lambda x: x[1])[0]
        
        # 4. Clean aggressively
        cleaned_text = cls._clean_text(best_candidate)
        
        if not cleaned_text.strip():
            return None
            
        # Extract just the first few sentences of the cleaned text to avoid confusing the small model 
        # with training details that usually appear at the end of the paragraph.
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        short_text = " ".join(sentences[:3])
            
        # 5 & 6 & 7. Summarize, Validate, Retry, Fallback
        prompt1 = f"In one sentence, explain what this AI model is designed to do based on this description:\n\n{short_text}"
        
        summary = cls._generate(prompt1, max_output_chars)
        
        if summary and cls._is_valid_summary(summary, model_id):
            return summary
            
        # Retry with stricter prompt
        logger.info("⚠️ First summary invalid, retrying with stricter prompt.")
        prompt2 = f"Summarize the main purpose of this AI model in one complete sentence:\n\n{cleaned_text}"
        summary2 = cls._generate(prompt2, max_output_chars)
        
        if summary2 and cls._is_valid_summary(summary2, model_id):
            return summary2
            
        # Fallback to cleaned text (first 1-2 sentences)
        logger.info("⚠️ Both LLM summaries invalid, falling back to cleaned extracted text.")
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        fallback_summary = " ".join(sentences[:2])
        if len(fallback_summary) > max_output_chars:
             return fallback_summary[:max_output_chars-3] + "..."
        return fallback_summary
