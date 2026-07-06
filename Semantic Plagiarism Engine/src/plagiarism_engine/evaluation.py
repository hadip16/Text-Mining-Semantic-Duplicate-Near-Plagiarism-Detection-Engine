"""Reusable evaluation functions for pairwise duplicate detection."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .minhash import MinHasher, jaccard
from .preprocessing import preprocess
from .simhash import hamming_distance, simhash_similarity, simhash_texts


@dataclass
class BinaryMetrics:
    method: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    avg_time_ms: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], method: str, threshold: float, elapsed_seconds: float) -> BinaryMetrics:
    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = max(1, len(y_true))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return BinaryMetrics(
        method=method,
        threshold=threshold,
        accuracy=(tp + tn) / total,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        avg_time_ms=(elapsed_seconds / total) * 1000.0,
    )


def evaluate_pairs(
    df: pd.DataFrame,
    text_col_a: str,
    text_col_b: str,
    label_col: str,
    shingle_size: int = 3,
    num_perm: int = 128,
    num_bands: int = 32,
    minhash_threshold: float = 0.25,
    simhash_threshold: float = 0.75,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate MinHash and SimHash on a labelled pair dataset.

    Returns metrics dataframe and row-level prediction dataframe.
    """
    mh = MinHasher(num_perm=num_perm)
    predictions: List[dict] = []
    y_true = df[label_col].astype(int).tolist()

    minhash_preds: List[int] = []
    simhash_preds: List[int] = []

    t0 = time.perf_counter()
    for idx, row in df.iterrows():
        text_a = str(row[text_col_a])
        text_b = str(row[text_col_b])
        s_a = preprocess(text_a, shingle_size=shingle_size)
        s_b = preprocess(text_b, shingle_size=shingle_size)
        sig_a = mh.signature(s_a)
        sig_b = mh.signature(s_b)
        exact_j = jaccard(s_a, s_b)
        minhash_sim = mh.similarity(sig_a, sig_b)
        minhash_pred = int(minhash_sim >= minhash_threshold)
        minhash_preds.append(minhash_pred)

        sim_a, sim_b = simhash_texts([text_a, text_b])
        hdist = hamming_distance(sim_a, sim_b)
        sim_sim = simhash_similarity(sim_a, sim_b)
        sim_pred = int(sim_sim >= simhash_threshold)
        simhash_preds.append(sim_pred)

        predictions.append(
            {
                "row_id": idx,
                "label": int(row[label_col]),
                "exact_jaccard": exact_j,
                "minhash_similarity": minhash_sim,
                "minhash_pred": minhash_pred,
                "simhash_similarity": sim_sim,
                "simhash_hamming_distance": hdist,
                "simhash_pred": sim_pred,
            }
        )
    elapsed = time.perf_counter() - t0

    metrics = [
        binary_metrics(y_true, minhash_preds, "MinHash_similarity_threshold", minhash_threshold, elapsed).as_dict(),
        binary_metrics(y_true, simhash_preds, "TFIDF_weighted_SimHash", simhash_threshold, elapsed).as_dict(),
    ]
    return pd.DataFrame(metrics), pd.DataFrame(predictions)
