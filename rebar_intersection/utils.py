"""Utility helpers."""

from __future__ import annotations

import numpy as np


def cosine_similarity(line1: np.ndarray, line2: np.ndarray) -> float:
    """Cosine similarity between two direction vectors."""
    dot_product = np.dot(line1, line2)
    norm1 = np.linalg.norm(line1)
    norm2 = np.linalg.norm(line2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))
