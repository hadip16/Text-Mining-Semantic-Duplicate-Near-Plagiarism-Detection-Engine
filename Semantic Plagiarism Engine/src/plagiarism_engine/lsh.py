"""Locality Sensitive Hashing for MinHash signatures."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


def _band_key(values: Sequence[int]) -> str:
    arr = np.asarray(values, dtype=np.uint64)
    payload = arr.tobytes()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


class LSHIndex:
    """Banding-based LSH index for MinHash signatures."""

    def __init__(self, num_bands: int = 32):
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        self.num_bands = int(num_bands)
        self.tables: List[Dict[str, List[str]]] = [defaultdict(list) for _ in range(self.num_bands)]
        self.rows_per_band: int | None = None
        self.doc_ids: Set[str] = set()

    def _check_signature(self, signature: Sequence[int]) -> np.ndarray:
        sig = np.asarray(signature, dtype=np.uint64)
        if sig.size % self.num_bands != 0:
            raise ValueError("signature length must be divisible by num_bands")
        rows = sig.size // self.num_bands
        if self.rows_per_band is None:
            self.rows_per_band = rows
        elif rows != self.rows_per_band:
            raise ValueError("all signatures must have the same length")
        return sig

    def add(self, doc_id: str, signature: Sequence[int]) -> None:
        sig = self._check_signature(signature)
        rows = self.rows_per_band or 0
        for band_idx in range(self.num_bands):
            start = band_idx * rows
            key = _band_key(sig[start : start + rows])
            self.tables[band_idx][key].append(doc_id)
        self.doc_ids.add(doc_id)

    def candidate_pairs(self) -> Set[Tuple[str, str]]:
        """Return all unordered doc pairs sharing at least one LSH bucket."""
        pairs: Set[Tuple[str, str]] = set()
        for table in self.tables:
            for bucket in table.values():
                if len(bucket) >= 2:
                    for a, b in combinations(sorted(set(bucket)), 2):
                        pairs.add((a, b))
        return pairs

    def comparison_reduction(self) -> dict:
        n = len(self.doc_ids)
        all_pairs = n * (n - 1) // 2
        candidates = len(self.candidate_pairs())
        reduction = 1.0 - (candidates / all_pairs) if all_pairs else 0.0
        return {"documents": n, "all_pairs": all_pairs, "candidate_pairs": candidates, "reduction_ratio": reduction}
