from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# --- Enums (from enhanced_extractor.py) ---
class DataSource(str, Enum):
    """Enumeration of data sources for provenance tracking"""
    HF_API = "huggingface_api"
    MODEL_CARD = "model_card_yaml"
    README_TEXT = "readme_text"
    CONFIG_FILE = "config_file"
    REPOSITORY_FILES = "repository_files"
    EXTERNAL_REFERENCE = "external_reference"
    INTELLIGENT_DEFAULT = "intelligent_default"
    PLACEHOLDER = "placeholder"
    REGISTRY_DRIVEN = "registry_driven"

class ConfidenceLevel(str, Enum):
    """Confidence levels for extracted data"""
    HIGH = "high"        # Direct API data, official sources
    MEDIUM = "medium"    # Inferred from reliable patterns
    LOW = "low"          # Weak inference or pattern matching
    NONE = "none"        # Placeholder values

# --- internal Models ---
class ExtractionResult(BaseModel):
    """Container for extraction results with full provenance"""
    value: Any
    source: DataSource
    confidence: ConfidenceLevel
    extraction_method: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    fallback_chain: List[str] = Field(default_factory=list)

    def __str__(self):
        return f"{self.value} (source: {self.source.value}, confidence: {self.confidence.value})"

# --- API Request Models ---
class GenerateRequest(BaseModel):
    model_id: str
    include_inference: bool = True
    use_best_practices: bool = True
    hf_token: Optional[str] = None

class BatchRequest(BaseModel):
    model_ids: List[str]
    include_inference: bool = True
    use_best_practices: bool = True
    hf_token: Optional[str] = None

# --- API Response Models ---
class AIBOMResponse(BaseModel):
    aibom: Dict[str, Any]
    model_id: str
    generated_at: str
    request_id: str
    download_url: str
    completeness_score: Optional[Dict[str, Any]] = None

class EnhancementReport(BaseModel):
    ai_enhanced: bool = False
    ai_model: Optional[str] = None
    original_score: Dict[str, Any]
    final_score: Dict[str, Any]
    improvement: float = 0
