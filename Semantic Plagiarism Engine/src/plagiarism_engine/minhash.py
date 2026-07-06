"""From-scratch MinHash implementation.

No datasketch or ready-made MinHash libraries are used. The implementation uses
stable cryptographic hashes for deterministic reproducibility and universal hash
functions to produce signature rows.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence, Set, Tuple

import numpy as np

MAX_HASH = (1 << 64) - 1
# Prime larger than 2**64. Used by the universal hash family.
_MERSENNE_PRIME = (1 << 89) - 1


def stable_hash(value: object, seed: int = 0) -> int:
    """Return a stable unsigned 64-bit hash for any hashable-looking value."""
    if isinstance(value, (tuple, list)):
        data = "\u241f".join(map(str, value)).encode("utf-8", errors="ignore")
    else:
        data = str(value).encode("utf-8", errors="ignore")
    h = hashlib.blake2b(data, digest_size=8, person=f"mh{seed:06d}"[:16].encode())
    return int.from_bytes(h.digest(), "big", signed=False)


class MinHasher:
    """Generate MinHash signatures with deterministic random coefficients."""

    def __init__(self, num_perm: int = 128, seed: int = 42):
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        self.num_perm = int(num_perm)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MAX_HASH, size=self.num_perm, dtype=np.uint64).astype(object)
        self.b = rng.integers(0, MAX_HASH, size=self.num_perm, dtype=np.uint64).astype(object)

    def signature(self, shingles: Iterable[Tuple[str, ...]]) -> np.ndarray:
        """Return a MinHash signature vector for a set/list of shingles."""
        shingles = list(shingles)
        sig = np.full(self.num_perm, MAX_HASH, dtype=object)
        if not shingles:
            return sig.astype(np.uint64)
        base_hashes = [stable_hash(s) for s in shingles]
        for x in base_hashes:
            values = ((self.a * x + self.b) % _MERSENNE_PRIME) & MAX_HASH
            sig = np.minimum(sig, values)
        return sig.astype(np.uint64)

    @staticmethod
    def similarity(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
        """Estimate Jaccard similarity from two signatures."""
        a = np.asarray(sig_a, dtype=np.uint64)
        b = np.asarray(sig_b, dtype=np.uint64)
        if a.shape != b.shape:
            raise ValueError("signatures must have the same shape")
        if a.size == 0:
            return 0.0
        return float(np.mean(a == b))


def jaccard(set_a: Set[Tuple[str, ...]], set_b: Set[Tuple[str, ...]]) -> float:
    """Exact Jaccard similarity for two shingle sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)
