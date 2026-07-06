"""Dataset and file-loading helpers for the CLI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    path: str | None = None


def read_text_file(path: str | Path) -> str:
    path = Path(path)
    for enc in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def load_documents_from_folder(folder: str | Path, suffixes: Sequence[str] = (".txt", ".md")) -> List[Document]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")
    suffixes = tuple(s.lower() for s in suffixes)
    docs: List[Document] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            docs.append(Document(doc_id=path.stem, text=read_text_file(path), path=str(path)))
    return docs


def load_pairs_csv(path: str | Path, text_col_a: str, text_col_b: str, label_col: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (text_col_a, text_col_b, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df[[text_col_a, text_col_b, label_col]].dropna(subset=[text_col_a, text_col_b]).copy()
    if limit is not None and limit > 0:
        df = df.head(limit)
    df[label_col] = df[label_col].astype(int)
    return df
