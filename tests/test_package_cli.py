import io
import unittest
from contextlib import redirect_stdout

import codetrace.__main__ as cli


class TestPackageCli(unittest.TestCase):
    def test_main_help_runs(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("CodeLens", stdout.getvalue())

    def test_main_dispatches_eval_command(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            cli.main(["eval", "--repo", ".", "--limit", "1"])
        text = stdout.getvalue()
        self.assertIn("CodeLens", text)


if __name__ == "__main__":
    unittest.main()
