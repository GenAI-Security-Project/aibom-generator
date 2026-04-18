from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# --- Enums (from enhanced_extractor.py) ---
class DataSource(StrEnum):
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


class ConfidenceLevel(StrEnum):
    """Confidence levels for extracted data"""

    HIGH = "high"  # Direct API data, official sources
    MEDIUM = "medium"  # Inferred from reliable patterns
    LOW = "low"  # Weak inference or pattern matching
    NONE = "none"  # Placeholder values


# --- internal Models ---
class ExtractionResult(BaseModel):
    """Container for extraction results with full provenance"""

    value: Any
    source: DataSource
    confidence: ConfidenceLevel
    extraction_method: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    fallback_chain: list[str] = Field(default_factory=list)

    def __str__(self):
        return f"{self.value} (source: {self.source.value}, confidence: {self.confidence.value})"


# --- API Request Models ---
class GenerateRequest(BaseModel):
    model_id: str
    include_inference: bool = True
    use_best_practices: bool = True
    hf_token: str | None = None


class BatchRequest(BaseModel):
    model_ids: list[str]
    include_inference: bool = True
    use_best_practices: bool = True
    hf_token: str | None = None


# --- API Response Models ---
class AIBOMResponse(BaseModel):
    aibom: dict[str, Any]
    model_id: str
    generated_at: str
    request_id: str
    download_url: str
    completeness_score: dict[str, Any] | None = None


class EnhancementReport(BaseModel):
    ai_enhanced: bool = False
    ai_model: str | None = None
    original_score: dict[str, Any]
    final_score: dict[str, Any]
    improvement: float = 0
