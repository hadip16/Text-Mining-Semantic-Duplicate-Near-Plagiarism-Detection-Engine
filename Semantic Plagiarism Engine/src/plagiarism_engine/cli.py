"""Command line interface for the semantic plagiarism engine."""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .dataset import load_documents_from_folder, load_pairs_csv, read_text_file
from .pan11 import build_pan11_pairs
from .evaluation import evaluate_pairs
from .lsh import LSHIndex
from .minhash import MinHasher, jaccard
from .preprocessing import preprocess, token_frequencies
from .simhash import compute_idf, hamming_distance, simhash_from_counts, simhash_similarity, simhash_texts


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def cmd_compare(args: argparse.Namespace) -> None:
    text_a = read_text_file(args.file_a)
    text_b = read_text_file(args.file_b)
    shingles_a = preprocess(text_a, shingle_size=args.shingle_size, remove_stopwords=not args.keep_stopwords)
    shingles_b = preprocess(text_b, shingle_size=args.shingle_size, remove_stopwords=not args.keep_stopwords)
    mh = MinHasher(num_perm=args.num_perm, seed=args.seed)
    sig_a = mh.signature(shingles_a)
    sig_b = mh.signature(shingles_b)
    sim_a, sim_b = simhash_texts([text_a, text_b])
    result = {
        "file_a": str(args.file_a),
        "file_b": str(args.file_b),
        "shingle_size": args.shingle_size,
        "num_perm": args.num_perm,
        "exact_jaccard": jaccard(shingles_a, shingles_b),
        "minhash_similarity": mh.similarity(sig_a, sig_b),
        "simhash_similarity": simhash_similarity(sim_a, sim_b),
        "simhash_hamming_distance": hamming_distance(sim_a, sim_b),
        "num_shingles_a": len(shingles_a),
        "num_shingles_b": len(shingles_b),
    }
    _ensure_parent(args.output)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_corpus(args: argparse.Namespace) -> None:
    docs = load_documents_from_folder(args.data)
    if len(docs) < 2:
        raise SystemExit("At least two .txt/.md documents are required for corpus search.")
    mh = MinHasher(num_perm=args.num_perm, seed=args.seed)
    shingle_sets = {doc.doc_id: preprocess(doc.text, args.shingle_size, remove_stopwords=not args.keep_stopwords) for doc in docs}
    signatures = {doc.doc_id: mh.signature(shingle_sets[doc.doc_id]) for doc in docs}

    lsh = LSHIndex(num_bands=args.num_bands)
    for doc in docs:
        lsh.add(doc.doc_id, signatures[doc.doc_id])
    candidate_pairs = lsh.candidate_pairs()

    # TF-IDF SimHash is computed once over the whole corpus.
    token_counts = {doc.doc_id: token_frequencies(doc.text, remove_stopwords=not args.keep_stopwords) for doc in docs}
    idf = compute_idf(list(token_counts.values()))
    simhashes = {doc_id: simhash_from_counts(counts, idf=idf) for doc_id, counts in token_counts.items()}

    rows: List[dict] = []
    t0 = time.perf_counter()
    for doc_id_a, doc_id_b in sorted(candidate_pairs):
        exact_j = jaccard(shingle_sets[doc_id_a], shingle_sets[doc_id_b])
        mh_sim = mh.similarity(signatures[doc_id_a], signatures[doc_id_b])
        sh_sim = simhash_similarity(simhashes[doc_id_a], simhashes[doc_id_b])
        hdist = hamming_distance(simhashes[doc_id_a], simhashes[doc_id_b])
        rows.append(
            {
                "doc_id_a": doc_id_a,
                "doc_id_b": doc_id_b,
                "exact_jaccard": exact_j,
                "minhash_similarity": mh_sim,
                "simhash_similarity": sh_sim,
                "simhash_hamming_distance": hdist,
                "passes_minhash_threshold": mh_sim >= args.threshold,
                "passes_simhash_threshold": sh_sim >= args.simhash_threshold,
            }
        )
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["minhash_similarity", "simhash_similarity"], ascending=False)
    _ensure_parent(args.output)
    df.to_csv(args.output, index=False)

    reduction = lsh.comparison_reduction()
    summary = {
        **reduction,
        "threshold": args.threshold,
        "simhash_threshold": args.simhash_threshold,
        "elapsed_seconds_after_lsh": elapsed,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_pairs(args: argparse.Namespace) -> None:
    df = load_pairs_csv(args.pairs, args.text_col_a, args.text_col_b, args.label_col, limit=args.limit)
    metrics, preds = evaluate_pairs(
        df,
        text_col_a=args.text_col_a,
        text_col_b=args.text_col_b,
        label_col=args.label_col,
        shingle_size=args.shingle_size,
        num_perm=args.num_perm,
        num_bands=args.num_bands,
        minhash_threshold=args.threshold,
        simhash_threshold=args.simhash_threshold,
    )
    _ensure_parent(args.output)
    metrics.to_csv(args.output, index=False)
    if args.predictions_output:
        _ensure_parent(args.predictions_output)
        preds.to_csv(args.predictions_output, index=False)
    print(metrics.to_string(index=False))


def cmd_prepare_pan11(args: argparse.Namespace) -> None:
    df = build_pan11_pairs(
        args.pan_root,
        args.output,
        max_positive=args.max_positive,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
        min_chars=args.min_chars,
        segment_mode=args.segment_mode,
    )
    summary = {
        "pan_root": str(args.pan_root),
        "output": str(args.output),
        "rows": int(len(df)),
        "positive_pairs": int((df["label"] == 1).sum()),
        "negative_pairs": int((df["label"] == 0).sum()),
        "segment_mode": args.segment_mode,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plagiarism_engine", description="Semantic duplicate and near-plagiarism detection CLI")
    parser.add_argument("--version", action="version", version="semantic-plagiarism-engine 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--shingle-size", type=int, default=3)
    common.add_argument("--num-perm", type=int, default=128)
    common.add_argument("--num-bands", type=int, default=32)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--keep-stopwords", action="store_true")

    p = sub.add_parser("compare", parents=[common], help="Compare two text files and write JSON scores")
    p.add_argument("--file-a", required=True)
    p.add_argument("--file-b", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("corpus", parents=[common], help="Find similar document candidates in a folder")
    p.add_argument("--data", required=True)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--simhash-threshold", type=float, default=0.75)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("pairs", parents=[common], help="Evaluate on a labelled pair CSV")
    p.add_argument("--pairs", required=True)
    p.add_argument("--text-col-a", required=True)
    p.add_argument("--text-col-b", required=True)
    p.add_argument("--label-col", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--simhash-threshold", type=float, default=0.75)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", default="outputs/pair_predictions.csv")
    p.set_defaults(func=cmd_pairs)

    p = sub.add_parser("prepare-pan11", help="Convert PAN-PC-11 XML annotations into a labelled pair CSV")
    p.add_argument("--pan-root", required=True, help="Extracted PAN-PC-11 folder, e.g. data/raw/pan11")
    p.add_argument("--output", default="data/processed/pan11_pairs.csv")
    p.add_argument("--max-positive", type=int, default=None, help="Optional cap on positive annotated pairs")
    p.add_argument("--negatives-per-positive", type=int, default=1)
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--segment-mode", choices=["annotated", "full"], default="annotated")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_prepare_pan11)
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
