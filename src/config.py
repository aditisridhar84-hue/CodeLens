"""Configuration module for CodeLens / CodeTrace AI."""

import os
import json
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
INDEXES_DIR = PROJECT_ROOT / "indexes"
RESULTS_DIR = PROJECT_ROOT / "results"
EVALUATION_DIR = RESULTS_DIR / "evaluation"

# Ensure runtime directories exist
for directory in [CONFIGS_DIR, DATA_DIR, INDEXES_DIR, RESULTS_DIR, EVALUATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model & Index Settings
# Using all-MiniLM-L6-v2 as the default lightweight, high-performance CPU embedding model
DEFAULT_EMBEDDING_MODEL = os.getenv("CODELENS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("CODELENS_BATCH_SIZE", "64"))

# Retrieval & Fusion Parameters
RRF_K = 60
CANDIDATE_POOL_SIZE = 50
FINAL_TOP_K = 10

# Deduplication Threshold
NEAR_DUPLICATE_THRESHOLD = 0.90

# Weights Path
WEIGHTS_PATH = CONFIGS_DIR / "weights.json"


def load_weights(weights_file: Path = WEIGHTS_PATH) -> dict:
    """Load intent-specific reranking weights."""
    if weights_file.exists():
        with open(weights_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "DATA_FLOW": {"semantic": 0.30, "bm25": 0.15, "symbol": 0.10, "operation": 0.20, "call": 0.20, "structure": 0.05},
        "DEPENDENCY": {"semantic": 0.25, "bm25": 0.15, "symbol": 0.15, "operation": 0.10, "call": 0.35, "structure": 0.00},
        "IMPLEMENTATION": {"semantic": 0.25, "bm25": 0.25, "symbol": 0.30, "operation": 0.10, "call": 0.05, "structure": 0.05},
        "BEHAVIOR": {"semantic": 0.35, "bm25": 0.15, "symbol": 0.10, "operation": 0.15, "call": 0.10, "structure": 0.15},
        "DEFAULT": {"semantic": 0.30, "bm25": 0.20, "symbol": 0.15, "operation": 0.15, "call": 0.10, "structure": 0.10}
    }
