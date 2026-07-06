"""TF-IDF weighted SimHash implementation from scratch."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .preprocessing import token_frequencies

BITS = 64


def stable_token_hash(token: str, bits: int = BITS) -> int:
    digest_size = max(8, bits // 8)
    h = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=digest_size, person=b"simhash64")
    return int.from_bytes(h.digest()[: bits // 8], "big", signed=False)


def compute_idf(doc_token_counts: Sequence[Counter[str]]) -> Dict[str, float]:
    """Smoothed IDF over a corpus of token Counters."""
    n_docs = len(doc_token_counts)
    df: Counter[str] = Counter()
    for counts in doc_token_counts:
        df.update(counts.keys())
    return {token: math.log((1 + n_docs) / (1 + freq)) + 1.0 for token, freq in df.items()}


def simhash_from_counts(counts: Mapping[str, int], idf: Mapping[str, float] | None = None, bits: int = BITS) -> int:
    """Build a weighted SimHash fingerprint as an integer."""
    if not counts:
        return 0
    idf = idf or {}
    vector = np.zeros(bits, dtype=float)
    total = float(sum(counts.values())) or 1.0
    for token, count in counts.items():
        tf = count / total
        weight = tf * idf.get(token, 1.0)
        h = stable_token_hash(token, bits=bits)
        for i in range(bits):
            bit = (h >> i) & 1
            vector[i] += weight if bit else -weight
    fingerprint = 0
    for i, value in enumerate(vector):
        if value >= 0:
            fingerprint |= 1 << i
    return fingerprint


def simhash_texts(texts: Sequence[str], bits: int = BITS) -> List[int]:
    counts = [token_frequencies(t) for t in texts]
    idf = compute_idf(counts)
    return [simhash_from_counts(c, idf=idf, bits=bits) for c in counts]


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return int((int(hash_a) ^ int(hash_b)).bit_count())


def simhash_similarity(hash_a: int, hash_b: int, bits: int = BITS) -> float:
    return 1.0 - hamming_distance(hash_a, hash_b) / bits
