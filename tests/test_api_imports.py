import unittest

from codetrace.api import app


class TestApiImports(unittest.TestCase):
    def test_fastapi_app_loads(self):
        self.assertIsNotNone(app)
        self.assertEqual(app.title, "CodeLens API")


if __name__ == "__main__":
    unittest.main()
