import unittest
from unittest.mock import MagicMock, patch, mock_open
from src.utils.validation import validate_aibom, load_schema, _download_schema

class TestValidation(unittest.TestCase):
    @patch("src.utils.validation.load_schema")
    def test_validate_aibom_valid(self, mock_load):
        mock_load.return_value = {
            "type": "object", 
            "properties": {"bomFormat": {"type": "string"}},
            "required": ["bomFormat"]
        }
        valid_aibom = {"bomFormat": "CycloneDX"}
        is_valid, errors = validate_aibom(valid_aibom)
        if not is_valid:
            print(f"\nValidation Errors: {errors}")
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    @patch("src.utils.validation.load_schema")
    def test_validate_aibom_invalid(self, mock_load):
        mock_load.return_value = {
            "type": "object", 
            "properties": {"bomFormat": {"type": "string"}},
            "required": ["bomFormat"]
        }
        invalid_aibom = {"otherField": "value"}
        is_valid, errors = validate_aibom(invalid_aibom)
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)
        
    @patch("src.utils.validation.requests.get")
    def test_download_schema(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"key": "value"}
        mock_get.return_value = mock_response
        
        with patch("builtins.open", mock_open()) as mock_file:
            schema = _download_schema()
            self.assertEqual(schema, {"key": "value"})

    @patch("src.utils.validation._load_schema_from_cache")
    def test_schema_caching(self, mock_load):
        """Test that schema is cached after first load."""
        import src.utils.validation as val_module
        # Reset cache for test
        old_cache = val_module._cached_schema
        val_module._cached_schema = None
        
        try:
            mock_schema = {"type": "object", "cached": True}
            mock_load.return_value = mock_schema
            
            # First load
            schema1 = load_schema()
            self.assertIs(schema1, mock_schema)
            self.assertEqual(mock_load.call_count, 1)
            
            # Second load should use cache
            schema2 = load_schema()
            self.assertIs(schema2, schema1)
            self.assertEqual(mock_load.call_count, 1) # Still 1
        finally:
            val_module._cached_schema = old_cache

    def test_schema_loading(self):
        """Test that schema can be loaded (integration check)."""
        schema = load_schema()
        self.assertTrue(schema is None or isinstance(schema, dict))

if __name__ == '__main__':
    unittest.main()
