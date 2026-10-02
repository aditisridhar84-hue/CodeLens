"""Query Understanding & Intent Routing for CodeLens / CodeTrace AI.

Extracts:
- Intent category: FUNCTION_LOOKUP, DATA_FLOW, CALL_FLOW, IMPLEMENTATION, VERSION_CHANGE, GENERAL_CODE_SEARCH
- Symbols / function names
- Extracted keywords
- Relationship requirements (before, after, calls, called_by)
- Language hints
- Version/history intent
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .chunker import split_identifier


@dataclass
class QueryAnalysis:
    original_query: str
    intent: str = "GENERAL_CODE_SEARCH"
    symbols: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    relationship_type: Optional[str] = None   # "before", "after", "calls", "called_by", "defines"
    target_version: Optional[str] = None     # "current", "all", "history", or specific sha
    language: Optional[str] = None           # "python", "javascript", "typescript"
    clean_query: str = ""


INTENT_PATTERNS = [
    (
        "VERSION_CHANGE",
        re.compile(
            r"\b(what changed|how did .* change|diff|difference|evolution|history|commit|previous version|new version|modified|renamed|v1|v2)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "CALL_FLOW",
        re.compile(
            r"\b(what happens before|what runs after|which function calls|who calls|called by|invoked by|caller|callee|call hierarchy|execution order|pipeline flow)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DATA_FLOW",
        re.compile(
            r"\b(how is .* normalized|preprocess|preprocessing|normalize|clean|sanitize|transform|parse input|data flow|input pipeline)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "FUNCTION_LOOKUP",
        re.compile(
            r"\b(where is .* handled|where is .* defined|which function|find function|locate|who defines|definition of)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "IMPLEMENTATION",
        re.compile(
            r"\b(show me the code|implementation of|how does .* work|how is .* implemented|algorithm for|method for)\b",
            re.IGNORECASE,
        ),
    ),
]

LANGUAGE_PATTERNS = [
    ("python", re.compile(r"\b(python|py)\b", re.IGNORECASE)),
    ("typescript", re.compile(r"\b(typescript|ts)\b", re.IGNORECASE)),
    ("javascript", re.compile(r"\b(javascript|js|node)\b", re.IGNORECASE)),
    ("java", re.compile(r"\b(java|jdk|spring)\b", re.IGNORECASE)),
    ("cpp", re.compile(r"\b(c\+\+|cpp|cxx|native)\b", re.IGNORECASE)),
    ("c", re.compile(r"\b(c language|plain c|c\b)\b", re.IGNORECASE)),
]


def extract_symbols(query: str) -> List[str]:
    """Extract code identifiers from natural language query."""
    symbols = []

    # 1. Backtick enclosed identifiers: `foo_bar`
    backticked = re.findall(r"`([A-Za-z0-9_]+)`", query)
    symbols.extend(backticked)

    # 2. Identifiers followed by parentheses: foo()
    fn_calls = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", query)
    symbols.extend(fn_calls)

    # 3. CamelCase or snake_case tokens
    words = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", query)
    stop_words = {
        "the", "is", "how", "what", "where", "which", "who", "show", "code", "function",
        "method", "class", "file", "before", "after", "input", "output", "data", "and",
        "or", "in", "to", "for", "with", "by", "from", "of", "handled", "defined", "runs",
        "changed", "calls", "called", "validates", "user", "main"
    }

    for w in words:
        w_lower = w.lower()
        if w_lower in stop_words:
            # Special case: 'main' is a very common function symbol
            if w_lower == "main" and "main" not in symbols:
                symbols.append("main")
            continue
        # If it's snake_case with underscores or CamelCase
        if "_" in w or (re.search(r"[a-z][A-Z]", w) and len(w) > 3):
            if w not in symbols:
                symbols.append(w)
        elif len(w) > 3 and (w.isupper() or any(c.isupper() for c in w[1:])):
            if w not in symbols:
                symbols.append(w)

    return list(dict.fromkeys(symbols))


def analyze_query(query: str) -> QueryAnalysis:
    """Analyze query intent, extract symbols, keywords, flow relationships, and version requirements."""
    q_str = query.strip()

    # Determine intent
    detected_intent = "GENERAL_CODE_SEARCH"
    for intent, pat in INTENT_PATTERNS:
        if pat.search(q_str):
            detected_intent = intent
            break

    # Relationship detection
    rel_type = None
    if re.search(r"\b(before|prior to|precedes)\b", q_str, re.IGNORECASE):
        rel_type = "before"
    elif re.search(r"\b(after|following|succeeds)\b", q_str, re.IGNORECASE):
        rel_type = "after"
    elif re.search(r"\b(calls|invokes)\b", q_str, re.IGNORECASE):
        rel_type = "calls"
    elif re.search(r"\b(called by|invoked by)\b", q_str, re.IGNORECASE):
        rel_type = "called_by"

    # Version requirement
    target_ver = "current"
    if detected_intent == "VERSION_CHANGE" or re.search(r"\b(all versions|history|evolution|across commits)\b", q_str, re.IGNORECASE):
        target_ver = "all"
    elif re.search(r"\b(previous|prior|older|v1)\b", q_str, re.IGNORECASE):
        target_ver = "history"

    # Language hint
    lang = None
    for l_name, l_pat in LANGUAGE_PATTERNS:
        if l_pat.search(q_str):
            lang = l_name
            break

    # Extract symbols and constituent keywords
    symbols = extract_symbols(q_str)

    # Keywords from all non-stopwords
    raw_words = re.findall(r"\b[A-Za-z0-9_]+\b", q_str.lower())
    ignore = {
        "a", "an", "the", "in", "on", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after", "above", "below",
        "to", "from", "up", "down", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "how", "what", "where", "which",
        "who", "whom", "this", "that", "these", "those", "can", "could", "should", "would",
        "show", "me", "find", "code"
    }
    keywords = [w for w in raw_words if w not in ignore and len(w) > 2]
    # Add split words from symbols
    for s in symbols:
        for sub in split_identifier(s):
            if sub not in keywords and len(sub) > 2:
                keywords.append(sub)

    return QueryAnalysis(
        original_query=query,
        intent=detected_intent,
        symbols=symbols,
        keywords=list(dict.fromkeys(keywords)),
        relationship_type=rel_type,
        target_version=target_ver,
        language=lang,
        clean_query=" ".join(keywords),
    )
