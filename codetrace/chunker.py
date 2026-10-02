"""Language-aware code parsing and chunking for CodeLens / CodeTrace AI.

Uses Tree-sitter for AST parsing of Python and JavaScript/TypeScript code,
with robust AST/regex fallbacks to guarantee 100% reliability.
Extracts rich structural metadata: calls, imports, docstrings, parent classes,
and code keywords.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Optional Tree-sitter imports
TREE_SITTER_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    PY_LANGUAGE = Language(tspython.language())
    py_parser = Parser(PY_LANGUAGE)
    TREE_SITTER_AVAILABLE = True
except Exception:
    py_parser = None

TS_JS_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript as tsjs
    JS_LANGUAGE = Language(tsjs.language())
    js_parser = Parser(JS_LANGUAGE)
    TS_JS_AVAILABLE = True
except Exception:
    js_parser = None


@dataclass
class CodeChunk:
    id: str = ""
    repo: str = ""
    commit: str = ""
    path: str = ""
    name: str = ""
    language: str = "python"
    start: int = 1
    end: int = 1
    code: str = ""
    code_hash: str = ""
    doc: str = ""
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    embedding_id: int = -1

    def __post_init__(self):
        if not self.code_hash and self.code:
            self.code_hash = hashlib.sha1(self.code.encode("utf-8", errors="ignore")).hexdigest()
        if not self.id:
            key = f"{self.repo}:{self.commit}:{self.path}:{self.name}:{self.start}-{self.end}:{self.code_hash}"
            self.id = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
        if not self.keywords and self.name:
            self.keywords = extract_identifier_keywords(self.name + " " + " ".join(self.calls[:5]))

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, CodeChunk):
            return self.id == other.id
        return False


def split_identifier(identifier: str) -> List[str]:
    """Split snake_case, camelCase, or PascalCase identifier into constituent words."""
    if not identifier:
        return []
    # Split on underscore, hyphen, or dot
    parts = re.split(r"[_\-\.]+", identifier)
    words = []
    for part in parts:
        # Split camelCase
        sub = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", part)
        if sub:
            words.extend([w.lower() for w in sub if len(w) > 1])
        elif part:
            words.append(part.lower())
    return [w for w in words if len(w) > 1]


def extract_identifier_keywords(text: str) -> List[str]:
    """Extract distinct keywords from identifiers found in text."""
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
    res = set()
    for tok in tokens:
        for w in split_identifier(tok):
            res.add(w)
    return sorted(res)


class AstCallVisitor(ast.NodeVisitor):
    """Visitor to collect function/method calls from a Python AST subtree."""
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        self.generic_visit(node)


class AstImportVisitor(ast.NodeVisitor):
    """Visitor to collect imported module/function names."""
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        for alias in node.names:
            self.imports.append(f"{mod}.{alias.name}" if mod else alias.name)
        self.generic_visit(node)


def chunk_python_source(src: str, path: str, commit: str = "HEAD", repo: str = "local") -> List[CodeChunk]:
    """Parse Python source code and return granular function, method, and class chunks.
    Uses AST parsing with rich metadata extraction and Tree-sitter fallback/support.
    """
    chunks: List[CodeChunk] = []
    lines = src.splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines == 0:
        return chunks

    # First collect module-level imports
    file_imports: List[str] = []
    try:
        root = ast.parse(src, filename=path)
        imp_vis = AstImportVisitor()
        imp_vis.visit(root)
        file_imports = list(dict.fromkeys(imp_vis.imports))
    except Exception:
        # Regex fallback for imports if AST fails to parse entire module
        for line in lines:
            m = re.match(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.,\s]+))", line)
            if m:
                file_imports.extend([s.strip() for s in (m.group(1) or m.group(2) or "").split(",") if s.strip()])

    # Try AST-based extraction
    try:
        root = ast.parse(src, filename=path)
        for node in root.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(_node_to_chunk(node, lines, path, commit, repo, file_imports, parent_class=None))
            elif isinstance(node, ast.ClassDef):
                # Also add class chunk itself
                class_chunk = _node_to_chunk(node, lines, path, commit, repo, file_imports, parent_class=None)
                chunks.append(class_chunk)
                # And methods inside class
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.append(_node_to_chunk(sub, lines, path, commit, repo, file_imports, parent_class=node.name))
        if chunks:
            return chunks
    except Exception:
        pass

    # If AST failed or returned no functions, use Tree-sitter if available
    if py_parser:
        try:
            tree = py_parser.parse(src.encode("utf-8", errors="ignore"))
            ts_chunks = _parse_tree_sitter_python(tree.root_node, src, lines, path, commit, repo, file_imports)
            if ts_chunks:
                return ts_chunks
        except Exception:
            pass

    # Fallback: regex-based function/class chunking
    fn_pattern = re.compile(r"^[ \t]*(async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
    matches = list(fn_pattern.finditer(src))
    if matches:
        for i, m in enumerate(matches):
            start_pos = m.start()
            start_line = src[:start_pos].count("\n") + 1
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
                end_line = src[:end_pos].count("\n")
            else:
                end_line = total_lines
            chunk_code = "".join(lines[start_line - 1 : end_line]).rstrip()
            sym = m.group(2)
            c = CodeChunk(
                repo=repo,
                commit=commit,
                path=path,
                name=sym,
                language="python",
                start=start_line,
                end=end_line,
                code=chunk_code,
                imports=file_imports,
                calls=_extract_calls_regex(chunk_code),
            )
            chunks.append(c)
        return chunks

    # Fallback: if no symbols found at all, create file-level chunk
    if src.strip():
        c = CodeChunk(
            repo=repo,
            commit=commit,
            path=path,
            name=Path(path).stem,
            language="python",
            start=1,
            end=total_lines,
            code=src.strip(),
            imports=file_imports,
            calls=_extract_calls_regex(src),
        )
        chunks.append(c)

    return chunks


def _node_to_chunk(
    node: ast.AST,
    lines: List[str],
    path: str,
    commit: str,
    repo: str,
    file_imports: List[str],
    parent_class: Optional[str] = None,
) -> CodeChunk:
    start_line = getattr(node, "lineno", 1)
    end_line = getattr(node, "end_lineno", start_line)
    code_text = "".join(lines[start_line - 1 : end_line]).rstrip()

    doc = ast.get_docstring(node) or ""
    call_vis = AstCallVisitor()
    call_vis.visit(node)
    calls = list(dict.fromkeys(call_vis.calls))
    name = getattr(node, "name", "anonymous")

    kw = extract_identifier_keywords(f"{name} {doc} " + " ".join(calls[:8]))

    return CodeChunk(
        repo=repo,
        commit=commit,
        path=path,
        name=name,
        language="python",
        start=start_line,
        end=end_line,
        code=code_text,
        doc=doc,
        imports=file_imports,
        calls=calls,
        parent_class=parent_class,
        keywords=kw,
    )


def _parse_tree_sitter_python(root_node, src: str, lines: List[str], path: str, commit: str, repo: str, imports: List[str]) -> List[CodeChunk]:
    chunks = []
    src_bytes = src.encode("utf-8", errors="ignore")

    def walk(node, parent_class=None):
        if node.type in ("function_definition", "class_definition"):
            name = ""
            for child in node.children:
                if child.type == "identifier":
                    name = src_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="ignore")
                    break
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            code_text = "".join(lines[start_line - 1 : end_line]).rstrip()
            calls = _extract_calls_regex(code_text)
            c = CodeChunk(
                repo=repo,
                commit=commit,
                path=path,
                name=name or "unnamed",
                language="python",
                start=start_line,
                end=end_line,
                code=code_text,
                imports=imports,
                calls=calls,
                parent_class=parent_class,
                keywords=extract_identifier_keywords(f"{name} " + " ".join(calls[:8])),
            )
            chunks.append(c)

            new_parent = name if node.type == "class_definition" else parent_class
            for child in node.children:
                if child.type == "block":
                    for sub in child.children:
                        walk(sub, new_parent)
            return

        for child in node.children:
            walk(child, parent_class)

    walk(root_node)
    return chunks


def _extract_calls_regex(code: str) -> List[str]:
    """Find called function names using regex heuristics."""
    found = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    reserved = {
        "def", "class", "if", "for", "while", "return", "with", "except", "lambda",
        "print", "len", "range", "str", "int", "float", "bool", "dict", "list", "set",
        "super", "isinstance", "type", "sum", "min", "max", "zip", "enumerate",
        "function", "const", "let", "var", "switch", "catch", "import", "export"
    }
    return [name for name in dict.fromkeys(found) if name not in reserved and len(name) > 1]


def chunk_js_source(src: str, path: str, commit: str = "HEAD", repo: str = "local") -> List[CodeChunk]:
    """Parse JavaScript/TypeScript code using Tree-sitter or regex fallback."""
    chunks: List[CodeChunk] = []
    lines = src.splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines == 0:
        return chunks

    # Extract imports
    imports = []
    for line in lines:
        m = re.match(r'^\s*import\s+.*from\s+[\'"](.*)[\'"]', line)
        if m:
            imports.append(m.group(1))

    if js_parser:
        try:
            tree = js_parser.parse(src.encode("utf-8", errors="ignore"))
            src_bytes = src.encode("utf-8", errors="ignore")

            def walk_js(node, parent_class=None):
                if node.type in ("function_declaration", "method_definition", "class_declaration", "arrow_function"):
                    name = ""
                    for child in node.children:
                        if child.type in ("identifier", "property_identifier"):
                            name = src_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="ignore")
                            break
                    if not name and node.parent and node.parent.type == "variable_declarator":
                        for c in node.parent.children:
                            if c.type == "identifier":
                                name = src_bytes[c.start_byte : c.end_byte].decode("utf-8", errors="ignore")
                                break
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    code_text = "".join(lines[start_line - 1 : end_line]).rstrip()
                    calls = _extract_calls_regex(code_text)
                    chunks.append(
                        CodeChunk(
                            repo=repo,
                            commit=commit,
                            path=path,
                            name=name or "anonymous",
                            language="javascript",
                            start=start_line,
                            end=end_line,
                            code=code_text,
                            imports=imports,
                            calls=calls,
                            parent_class=parent_class,
                            keywords=extract_identifier_keywords(f"{name} " + " ".join(calls[:8])),
                        )
                    )
                    new_parent = name if node.type == "class_declaration" else parent_class
                    for child in node.children:
                        walk_js(child, new_parent)
                    return

                for child in node.children:
                    walk_js(child, parent_class)

            walk_js(tree.root_node)
            if chunks:
                return chunks
        except Exception:
            pass

    # Regex fallback for JS/TS
    js_fn = re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function\s+([A-Za-z0-9_$]+)|class\s+([A-Za-z0-9_$]+)|const\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?\()",
        re.MULTILINE,
    )
    matches = list(js_fn.finditer(src))
    if matches:
        for i, m in enumerate(matches):
            sym = m.group(1) or m.group(2) or m.group(3) or "anonymous"
            start_pos = m.start()
            start_line = src[:start_pos].count("\n") + 1
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
                end_line = src[:end_pos].count("\n")
            else:
                end_line = total_lines
            chunk_code = "".join(lines[start_line - 1 : end_line]).rstrip()
            chunks.append(
                CodeChunk(
                    repo=repo,
                    commit=commit,
                    path=path,
                    name=sym,
                    language="javascript",
                    start=start_line,
                    end=end_line,
                    code=chunk_code,
                    imports=imports,
                    calls=_extract_calls_regex(chunk_code),
                    keywords=extract_identifier_keywords(sym),
                )
            )
        return chunks

    if src.strip():
        chunks.append(
            CodeChunk(
                repo=repo,
                commit=commit,
                path=path,
                name=Path(path).stem,
                language="javascript",
                start=1,
                end=total_lines,
                code=src.strip(),
                imports=imports,
                calls=_extract_calls_regex(src),
            )
        )
    return chunks


def _find_matching_brace(src: str, start_index: int) -> int:
    """Find the closing brace for a function or class definition."""
    depth = 0
    quote = None
    escape = False

    for idx in range(start_index, len(src)):
        ch = src[idx]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue

        if ch in ('"', "'"):
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx

    return len(src) - 1


def chunk_c_family_source(src: str, path: str, language: str = "c", commit: str = "HEAD", repo: str = "local") -> List[CodeChunk]:
    """Chunk Java, C, and C++ code with a lightweight brace-aware parser."""
    lines = src.splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines == 0:
        return []

    imports: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#include") or stripped.startswith("import "):
            imports.append(stripped)

    if language and language.lower().startswith("c++"):
        lang_name = "cpp"
    elif language and language.lower() == "java":
        lang_name = "java"
    else:
        lang_name = "c"

    matches: List[tuple[int, int, str]] = []
    function_pattern = re.compile(
        r"(?:^|[\s;{}])(?:public|private|protected|static|virtual|inline|friend|constexpr|const|final|override|typename|template\s*<[^>]+>\s*)*" \
        r"[A-Za-z_:<>\[\]\*&\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*(?:const\s*)?\s*\{",
        re.MULTILINE,
    )
    for match in function_pattern.finditer(src):
        name = match.group(1)
        if not name or name in {"if", "for", "while", "switch", "catch", "return", "sizeof"}:
            continue
        start = match.start()
        end = _find_matching_brace(src, src.find("{", start))
        if end <= start:
            continue
        start_line = src[:start].count("\n") + 1
        end_line = src[:end].count("\n") + 1
        matches.append((start_line, end_line, name))

    class_pattern = re.compile(r"\b(?:class|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.MULTILINE)
    for match in class_pattern.finditer(src):
        name = match.group(1)
        start = match.start()
        brace_idx = src.find("{", start)
        if brace_idx == -1:
            continue
        end = _find_matching_brace(src, brace_idx)
        if end <= start:
            continue
        start_line = src[:start].count("\n") + 1
        end_line = src[:end].count("\n") + 1
        matches.append((start_line, end_line, name))

    if not matches and src.strip():
        return [
            CodeChunk(
                repo=repo,
                commit=commit,
                path=path,
                name=Path(path).stem,
                language=lang_name,
                start=1,
                end=total_lines,
                code=src.strip(),
                imports=imports,
                calls=_extract_calls_regex(src),
            )
        ]

    chunks: List[CodeChunk] = []
    seen: set[str] = set()
    for start_line, end_line, name in matches:
        if name in seen:
            continue
        seen.add(name)
        code_text = "".join(lines[start_line - 1 : end_line]).rstrip()
        chunks.append(
            CodeChunk(
                repo=repo,
                commit=commit,
                path=path,
                name=name,
                language=lang_name,
                start=start_line,
                end=end_line,
                code=code_text,
                imports=imports,
                calls=_extract_calls_regex(code_text),
                keywords=extract_identifier_keywords(f"{name} " + " ".join(_extract_calls_regex(code_text)[:8])),
            )
        )

    if chunks:
        return chunks

    return [
        CodeChunk(
            repo=repo,
            commit=commit,
            path=path,
            name=Path(path).stem,
            language=lang_name,
            start=1,
            end=total_lines,
            code=src.strip(),
            imports=imports,
            calls=_extract_calls_regex(src),
        )
    ]


def chunk_source_code(src: str, path: str, language: Optional[str] = None, commit: str = "HEAD", repo: str = "local") -> List[CodeChunk]:
    """Dispatch chunking to the appropriate language parser."""
    p_lower = path.lower()
    if p_lower.endswith(".py") or language == "python":
        return chunk_python_source(src, path, commit=commit, repo=repo)
    elif p_lower.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs")) or language in ("javascript", "typescript"):
        return chunk_js_source(src, path, commit=commit, repo=repo)
    elif p_lower.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".java")) or language in ("c", "cpp", "c++", "java"):
        detected = language or ("java" if p_lower.endswith(".java") else "c")
        if detected.lower().startswith("c++") or detected.lower() in {"cpp", "cxx"}:
            detected = "cpp"
        return chunk_c_family_source(src, path, language=detected, commit=commit, repo=repo)
    else:
        # Generic fallback
        return chunk_python_source(src, path, commit=commit, repo=repo)


def chunk_repo(
    repo_path: str,
    extensions: tuple = (".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".java"),
    commit: str = "WORKTREE",
    repo_name: str = "local",
) -> List[CodeChunk]:
    """Scan a local directory and chunk all recognized code files."""
    chunks: List[CodeChunk] = []
    root = Path(repo_path).resolve()
    skip_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", ".next", ".cache"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for fn in filenames:
            if fn.startswith(".") or not any(fn.lower().endswith(ext) for ext in extensions):
                continue
            full_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(full_path, root).replace("\\", "/")
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                file_chunks = chunk_source_code(content, rel_path, commit=commit, repo=repo_name)
                chunks.extend(file_chunks)
            except Exception:
                continue
    return chunks
