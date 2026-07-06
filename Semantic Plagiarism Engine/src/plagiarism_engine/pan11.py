"""Utilities for preparing PAN-PC-11 data for pairwise evaluation.

PAN-PC-11 external plagiarism detection data is distributed as raw documents
plus XML annotations. The pairwise evaluator in this project expects a CSV with
plain text columns and a binary label. This module converts PAN XML annotations
into such a CSV without committing the large raw dataset.
"""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .dataset import read_text_file


@dataclass(frozen=True)
class PanFeature:
    suspicious_reference: str
    source_reference: str
    suspicious_path: Path
    source_path: Path
    this_offset: int
    this_length: int
    source_offset: int
    source_length: int
    plagiarism_type: str | None = None
    obfuscation: str | None = None


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _build_text_index(root: Path) -> Dict[str, List[Path]]:
    """Index text-like files by basename for fast XML reference resolution."""
    suffixes = {".txt", ".text"}
    index: Dict[str, List[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index.setdefault(path.name, []).append(path)
    return index


def _prefer_path(paths: Sequence[Path], hints: Sequence[str]) -> Optional[Path]:
    if not paths:
        return None
    lowered_hints = [h.lower() for h in hints]
    for path in paths:
        parent_text = str(path).lower()
        if any(h in parent_text for h in lowered_hints):
            return path
    return paths[0]


def _resolve_text_reference(reference: str, index: Dict[str, List[Path]], root: Path, hints: Sequence[str]) -> Optional[Path]:
    """Resolve PAN XML document references to actual local files."""
    if not reference:
        return None
    ref = reference.replace("\\", "/")
    basename = Path(ref).name

    # Most PAN references are just basenames. Try the basename index first.
    if basename in index:
        return _prefer_path(index[basename], hints)

    # Fall back to direct relative path lookup.
    direct = root / ref
    if direct.exists() and direct.is_file():
        return direct

    # Some annotations omit extensions or use slightly different paths.
    candidates = list(root.rglob(basename))
    return _prefer_path(candidates, hints)


def _slice_text(text: str, offset: int, length: int, min_chars: int = 1) -> str:
    if offset < 0:
        offset = 0
    if length <= 0:
        return text.strip()
    segment = text[offset : offset + length].strip()
    if len(segment) < min_chars:
        return text.strip()
    return segment


def iter_pan11_features(pan_root: str | Path) -> List[PanFeature]:
    """Parse PAN XML annotations and return plagiarism features.

    The function is intentionally tolerant because PAN archives can be extracted
    with slightly different directory names. It supports the common PAN XML
    pattern where the root document has a `reference` attribute and child
    `<feature name="plagiarism" ... source_reference="..." />` entries.
    """
    root = Path(pan_root)
    if not root.exists():
        raise FileNotFoundError(f"PAN root not found: {root}")

    text_index = _build_text_index(root)
    features: List[PanFeature] = []

    for xml_path in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        xml_root = tree.getroot()
        suspicious_reference = xml_root.attrib.get("reference") or f"{xml_path.stem}.txt"
        suspicious_path = _resolve_text_reference(suspicious_reference, text_index, root, hints=("suspicious",))
        if suspicious_path is None:
            continue

        for feat in xml_root.iter("feature"):
            if feat.attrib.get("name") != "plagiarism":
                continue
            source_reference = feat.attrib.get("source_reference") or feat.attrib.get("source") or ""
            if not source_reference:
                continue
            source_path = _resolve_text_reference(source_reference, text_index, root, hints=("source",))
            if source_path is None:
                continue
            features.append(
                PanFeature(
                    suspicious_reference=suspicious_reference,
                    source_reference=source_reference,
                    suspicious_path=suspicious_path,
                    source_path=source_path,
                    this_offset=_safe_int(feat.attrib.get("this_offset")),
                    this_length=_safe_int(feat.attrib.get("this_length")),
                    source_offset=_safe_int(feat.attrib.get("source_offset")),
                    source_length=_safe_int(feat.attrib.get("source_length")),
                    plagiarism_type=feat.attrib.get("type"),
                    obfuscation=feat.attrib.get("obfuscation"),
                )
            )
    return features


def build_pan11_pairs(
    pan_root: str | Path,
    output: str | Path,
    *,
    max_positive: int | None = None,
    negatives_per_positive: int = 1,
    seed: int = 42,
    min_chars: int = 80,
    segment_mode: str = "annotated",
) -> pd.DataFrame:
    """Build a labelled pair CSV from PAN-PC-11 annotations.

    Parameters
    ----------
    pan_root:
        Extracted PAN-PC-11 root folder, usually `data/raw/pan11` or the
        extracted `external-detection-corpus` folder.
    output:
        CSV path to write, usually `data/processed/pan11_pairs.csv`.
    max_positive:
        Optional cap on positive annotations for quick experiments.
    negatives_per_positive:
        Number of random negative examples sampled for each positive feature.
    seed:
        Deterministic random seed for reproducibility.
    min_chars:
        Drop pairs where either side is shorter than this after slicing.
    segment_mode:
        `annotated` uses annotated plagiarism/source passages. `full` uses the
        full suspicious and source documents.
    """
    if negatives_per_positive < 0:
        raise ValueError("negatives_per_positive must be non-negative")
    if segment_mode not in {"annotated", "full"}:
        raise ValueError("segment_mode must be either 'annotated' or 'full'")

    features = iter_pan11_features(pan_root)
    if not features:
        raise ValueError(
            "No PAN plagiarism XML features were found. Check that --pan-root points to the extracted PAN-PC-11 folder."
        )
    if max_positive is not None and max_positive > 0:
        features = features[:max_positive]

    rng = random.Random(seed)
    text_index = _build_text_index(Path(pan_root))
    indexed_paths = {p for paths in text_index.values() for p in paths}
    likely_source_paths = {p for p in indexed_paths if "source" in str(p).lower()}
    all_source_paths = sorted(likely_source_paths | {feature.source_path for feature in features})
    rows: List[dict] = []
    seen = set()

    def positive_texts(feature: PanFeature) -> Tuple[str, str]:
        sus_text = read_text_file(feature.suspicious_path)
        src_text = read_text_file(feature.source_path)
        if segment_mode == "full":
            return sus_text.strip(), src_text.strip()
        return (
            _slice_text(sus_text, feature.this_offset, feature.this_length, min_chars=min_chars),
            _slice_text(src_text, feature.source_offset, feature.source_length, min_chars=min_chars),
        )

    for feature in features:
        text_a, text_b = positive_texts(feature)
        if len(text_a) >= min_chars and len(text_b) >= min_chars:
            key = (feature.suspicious_path.name, feature.source_path.name, feature.this_offset, feature.source_offset, 1)
            if key not in seen:
                rows.append(
                    {
                        "text_a": text_a,
                        "text_b": text_b,
                        "label": 1,
                        "suspicious_file": feature.suspicious_path.name,
                        "source_file": feature.source_path.name,
                        "this_offset": feature.this_offset,
                        "this_length": feature.this_length,
                        "source_offset": feature.source_offset,
                        "source_length": feature.source_length,
                        "plagiarism_type": feature.plagiarism_type or "",
                        "obfuscation": feature.obfuscation or "",
                    }
                )
                seen.add(key)

        # Negative examples: same suspicious passage, unrelated source document.
        if negatives_per_positive and all_source_paths:
            other_sources = [p for p in all_source_paths if p != feature.source_path]
            if not other_sources:
                continue
            k = min(negatives_per_positive, len(other_sources))
            for neg_source in rng.sample(other_sources, k=k):
                sus_full = read_text_file(feature.suspicious_path)
                neg_full = read_text_file(neg_source)
                if segment_mode == "full":
                    neg_a, neg_b = sus_full.strip(), neg_full.strip()
                else:
                    neg_a = _slice_text(sus_full, feature.this_offset, feature.this_length, min_chars=min_chars)
                    # Use a deterministic random slice from the negative document.
                    if len(neg_full) > max(min_chars, feature.source_length):
                        max_start = max(0, len(neg_full) - max(min_chars, feature.source_length))
                        start = rng.randint(0, max_start) if max_start > 0 else 0
                        neg_b = neg_full[start : start + max(min_chars, feature.source_length)].strip()
                    else:
                        neg_b = neg_full.strip()
                if len(neg_a) < min_chars or len(neg_b) < min_chars:
                    continue
                key = (feature.suspicious_path.name, neg_source.name, feature.this_offset, 0, 0)
                if key in seen:
                    continue
                rows.append(
                    {
                        "text_a": neg_a,
                        "text_b": neg_b,
                        "label": 0,
                        "suspicious_file": feature.suspicious_path.name,
                        "source_file": neg_source.name,
                        "this_offset": feature.this_offset,
                        "this_length": feature.this_length,
                        "source_offset": "",
                        "source_length": "",
                        "plagiarism_type": "negative_sample",
                        "obfuscation": "",
                    }
                )
                seen.add(key)

    if not rows:
        raise ValueError("PAN annotations were found, but all generated pairs were filtered out. Try lowering --min-chars.")

    df = pd.DataFrame(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df
