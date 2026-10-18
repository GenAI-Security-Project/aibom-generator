from datetime import datetime

from src.models.schemas import ConfidenceLevel, DataSource, ExtractionResult


def test_extraction_timestamp_is_timezone_aware_utc():
    result = ExtractionResult(
        value="value",
        source=DataSource.HF_API,
        confidence=ConfidenceLevel.HIGH,
        extraction_method="test",
    )

    timestamp = datetime.fromisoformat(result.timestamp)
    assert timestamp.utcoffset() is not None
    assert timestamp.utcoffset().total_seconds() == 0
