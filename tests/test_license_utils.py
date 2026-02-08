import unittest
from src.utils.license_utils import normalize_license_id, get_license_url

class TestLicenseUtils(unittest.TestCase):
    def test_normalize_license_id(self):
        self.assertEqual(normalize_license_id("Apache-2.0"), "Apache-2.0")
        self.assertEqual(normalize_license_id("apache-2.0"), "Apache-2.0")
        self.assertEqual(normalize_license_id("mit"), "MIT")
        self.assertEqual(normalize_license_id("MIT License"), "MIT")
        self.assertIsNone(normalize_license_id(None))
        
    def test_get_license_url(self):
        self.assertEqual(get_license_url("Apache-2.0"), "https://www.apache.org/licenses/LICENSE-2.0.txt")
        self.assertEqual(get_license_url("MIT"), "https://opensource.org/licenses/MIT")
        self.assertTrue("spdx.org" in get_license_url("unknown-license"))

if __name__ == '__main__':
    unittest.main()
