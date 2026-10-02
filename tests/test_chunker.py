import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from codetrace.chunker import chunk_source_code, chunk_python_source, chunk_js_source
from codetrace.git_tracker import GitTracker, normalize_git_sha
from codetrace.versions import HistoryIndex
import git

class TestChunker(unittest.TestCase):
    def test_python_parsing(self):
        code = '''import os
from math import sqrt

class DataProcessor:
    def normalize_input(self, raw_data):
        """Normalize raw input data before feeding to model."""
        cleaned = clean(raw_data)
        return validate(cleaned)

def main():
    dp = DataProcessor()
    dp.normalize_input([1, 2, 3])
'''
        chunks = chunk_python_source(code, "src/processor.py", commit="c123", repo="test_repo")
        self.assertGreaterEqual(len(chunks), 2)
        names = [c.name for c in chunks]
        self.assertIn("normalize_input", names)
        self.assertIn("main", names)
        norm_chunk = [c for c in chunks if c.name == "normalize_input"][0]
        self.assertEqual(norm_chunk.parent_class, "DataProcessor")
        self.assertIn("clean", norm_chunk.calls)
        self.assertIn("validate", norm_chunk.calls)
        self.assertIn("normalize", norm_chunk.keywords)
        self.assertIn("input", norm_chunk.keywords)

    def test_javascript_parsing(self):
        js_code = '''import { verify } from './auth';

export function authenticateUser(token) {
    const verified = verify(token);
    return loadUser(verified);
}

class SessionManager {
    logout(userId) {
        clearSession(userId);
    }
}
'''
        chunks = chunk_js_source(js_code, "src/auth.js", commit="c123", repo="test_repo")
        self.assertGreaterEqual(len(chunks), 2)
        names = [c.name for c in chunks]
        self.assertIn("authenticateUser", names)
        auth_chunk = [c for c in chunks if c.name == "authenticateUser"][0]
        self.assertIn("verify", auth_chunk.calls)
        self.assertIn("loadUser", auth_chunk.calls)

    def test_c_family_and_java_parsing(self):
        cpp_code = '''
class Logger {
public:
    void logMessage(const std::string& msg) {
        print(msg);
    }
};

int main() {
    Logger logger;
    logger.logMessage("hello");
    return 0;
}
'''
        cpp_chunks = chunk_source_code(cpp_code, "src/logger.cpp", language="cpp", commit="c123", repo="test_repo")
        self.assertTrue(any(c.name == "logMessage" for c in cpp_chunks))
        self.assertTrue(any(c.name == "main" for c in cpp_chunks))

        java_code = '''
class UserService {
    public String getName(int id) {
        return lookup(id);
    }
}
'''
        java_chunks = chunk_source_code(java_code, "src/UserService.java", language="java", commit="c123", repo="test_repo")
        self.assertTrue(any(c.name == "getName" for c in java_chunks))
        get_name = [c for c in java_chunks if c.name == "getName"][0]
        self.assertIn("lookup", get_name.calls)

        c_code = '''
int add(int a, int b) {
    return a + b;
}
'''
        c_chunks = chunk_source_code(c_code, "src/math.c", language="c", commit="c123", repo="test_repo")
        self.assertTrue(any(c.name == "add" for c in c_chunks))

    def test_git_sha_normalization(self):
        self.assertEqual(normalize_git_sha(b"abc123def456"), "abc123def456")
        self.assertEqual(normalize_git_sha("abc123def456"), "abc123def456")
        self.assertEqual(normalize_git_sha("b'abc123def456'"), "abc123def456")
        self.assertIsNone(normalize_git_sha(None))

    def test_incremental_index_falls_back_to_head_for_missing_sha(self):
        with TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            repo = git.Repo.init(repo_path)
            source_file = repo_path / "module.py"
            source_file.write_text("def main():\n    return True\n", encoding="utf-8")
            repo.index.add([str(source_file)])
            head_commit = repo.index.commit("Add module")

            history = HistoryIndex(
                chunks=[],
                presence=[],
                commits=[{"sha": "WORKTREE", "date": "", "author": "", "message": "snapshot"}],
            )
            tracker = GitTracker(base_data_dir=str(repo_path / "repos"))

            with patch("codetrace.git_tracker.save_sqlite"):
                updated_history, _, stats = tracker.incremental_index(
                    history,
                    pipe=object(),
                    repo_path=str(repo_path),
                    new_sha="b'5d8a58b33a2f569034ea395ca7090922eb4f33ad'",
                )

        self.assertTrue(stats["updated"])
        self.assertTrue(stats["used_head_fallback"])
        self.assertEqual(updated_history.latest_commit["sha"], head_commit.hexsha)

if __name__ == "__main__":
    unittest.main()
