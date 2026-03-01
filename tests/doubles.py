class FakeModelFileExtractor:
    def __init__(self, can=True, metadata=None):
        self._can = can
        self._metadata = metadata or {}

    def can_extract(self, model_id):
        return self._can

    def extract_metadata(self, model_id):
        return self._metadata


class FailingModelFileExtractor:
    def can_extract(self, model_id):
        raise RuntimeError("boom")

    def extract_metadata(self, model_id):
        raise RuntimeError("boom")
