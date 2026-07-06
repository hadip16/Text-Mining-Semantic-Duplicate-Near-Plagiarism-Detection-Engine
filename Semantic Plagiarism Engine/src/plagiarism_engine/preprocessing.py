"""Text normalization, tokenization, and word-shingling utilities.

The functions are intentionally small and dependency-light so the full pipeline
is easy to inspect and reproduce from the command line.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence, Set, Tuple

# Compact stopword lists. They remove extremely common function words while
# keeping enough lexical content for short educational examples.
EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "will", "with", "you", "your", "we", "our", "can",
}

FA_STOPWORDS = {
    "از", "است", "این", "آن", "به", "با", "برای", "در", "را", "که", "و", "یا",
    "یک", "می", "ها", "های", "بود", "شد", "شود", "کرد", "کند", "تا", "هم", "اما",
}

_DEFAULT_STOPWORDS = EN_STOPWORDS | FA_STOPWORDS
_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", flags=re.UNICODE)

_TRANSLATION_TABLE = str.maketrans({
    "ي": "ی",
    "ك": "ک",
    "ۀ": "ه",
    "ة": "ه",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "‌": " ",  # ZWNJ -> space for stable tokenization
})


def normalize_text(text: str) -> str:
    """Return a deterministic normalized representation of *text*.

    The function handles Latin and Persian/Arabic scripts: Unicode NFKC,
    lowercasing, Persian/Arabic character unification, punctuation removal,
    and whitespace compaction.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(_TRANSLATION_TABLE)
    text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    return " ".join(tokens)


def tokenize(text: str, remove_stopwords: bool = True, min_token_len: int = 1) -> List[str]:
    """Normalize and split text into tokens.

    Empty, whitespace-only, or non-textual documents safely return an empty list.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens = normalized.split()
    tokens = [t for t in tokens if len(t) >= min_token_len]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _DEFAULT_STOPWORDS]
    return tokens


def word_shingles(tokens: Sequence[str], shingle_size: int = 3) -> Set[Tuple[str, ...]]:
    """Convert tokens to word shingles.

    For very short documents, the whole token sequence is kept as one shingle.
    Empty documents return an empty set.
    """
    if shingle_size <= 0:
        raise ValueError("shingle_size must be positive")
    tokens = list(tokens)
    if not tokens:
        return set()
    if len(tokens) < shingle_size:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + shingle_size]) for i in range(len(tokens) - shingle_size + 1)}


def preprocess(text: str, shingle_size: int = 3, remove_stopwords: bool = True) -> Set[Tuple[str, ...]]:
    """Complete preprocessing path: text -> tokens -> word shingles."""
    return word_shingles(tokenize(text, remove_stopwords=remove_stopwords), shingle_size=shingle_size)


def token_frequencies(text: str, remove_stopwords: bool = True) -> Counter[str]:
    """Return token counts for TF-IDF based SimHash."""
    return Counter(tokenize(text, remove_stopwords=remove_stopwords))


def shingle_to_string(shingle: Iterable[str]) -> str:
    return " ".join(shingle)
