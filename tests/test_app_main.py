import shutil
import tempfile
import unittest
from pathlib import Path

import app


class TestAppMain(unittest.TestCase):
    def test_main_function_exists(self):
        self.assertTrue(callable(app.main))

    def test_demo_repo_helper_creates_repository(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="codelens_demo_"))
        repo_path = tmpdir / "demo_repo"
        result = app.ensure_repository_input(str(repo_path))
        self.assertEqual(result, str(repo_path))
        self.assertTrue(repo_path.exists())
        self.assertTrue((repo_path / "src").exists())
        self.assertTrue(any(repo_path.rglob("*.py")))
        shutil.rmtree(tmpdir)

    def test_create_demo_git_repo_reuses_existing_git_repo(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="codelens_existing_"))
        repo_path = tmpdir / "existing_repo"
        repo_path.mkdir(parents=True, exist_ok=True)
        import git
        repo = git.Repo.init(str(repo_path))
        (repo_path / "sample.py").write_text("def hi():\n    return 1\n", encoding="utf-8")
        repo.index.add(["sample.py"])
        repo.index.commit("init")

        result = app.create_demo_git_repo(str(repo_path))
        self.assertEqual(result, str(repo_path))
        self.assertTrue((repo_path / ".git").exists())
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
