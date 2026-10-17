import unittest
import threading
from unittest.mock import patch, MagicMock
from src.utils.analytics import log_sbom_generation, get_sbom_count, _log_queue
import src.utils.analytics

class AnalyticsTests(unittest.TestCase):
    @patch('src.utils.analytics.HF_TOKEN', "dummy")
    @patch('src.utils.analytics.load_dataset')
    @patch('src.utils.analytics.Dataset.push_to_hub')
    def test_log_sbom_generation(self, mock_push, mock_load):
        # Enqueue item
        log_sbom_generation("model-test")

        # Give worker thread a moment to process or join queue
        _log_queue.join()

        # Need to give a bit of time for the push to actually finish since task_done is called
        # *after* we get from queue but *before* push_to_hub finishes if batching gets it.
        # Actually in our code task_done is called BEFORE push_batch to ensure we can fetch more.
        # So we wait a tiny bit to let push_batch run.
        import time
        time.sleep(0.5)

        # Verify push was called
        mock_push.assert_called()

    @patch('src.utils.analytics.HF_TOKEN', "dummy")
    @patch('src.utils.analytics.load_dataset')
    def test_get_sbom_count(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 42
        mock_load.return_value = mock_ds

        count = get_sbom_count()
        # count is len(ds) + queue.qsize()
        expected = str(42 + _log_queue.qsize())
        self.assertEqual(count, expected)

if __name__ == "__main__":
    unittest.main()
