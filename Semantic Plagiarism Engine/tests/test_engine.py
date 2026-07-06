from pathlib import Path

from plagiarism_engine.lsh import LSHIndex
from plagiarism_engine.minhash import MinHasher, jaccard
from plagiarism_engine.preprocessing import preprocess, tokenize
from plagiarism_engine.simhash import hamming_distance, simhash_similarity, simhash_texts


def test_preprocessing_handles_persian_and_english():
    tokens = tokenize("This is a TEST! اين يك متن آزمايشي است.")
    assert "test" in tokens
    assert "متن" in tokens
    assert "is" not in tokens


def test_exact_jaccard_identical_documents():
    s1 = preprocess("minhash detects similar documents", shingle_size=2)
    s2 = preprocess("minhash detects similar documents", shingle_size=2)
    assert jaccard(s1, s2) == 1.0


def test_minhash_identical_signatures_are_equal():
    shingles = preprocess("one two three four five", shingle_size=2)
    mh = MinHasher(num_perm=64, seed=123)
    sig1 = mh.signature(shingles)
    sig2 = mh.signature(shingles)
    assert mh.similarity(sig1, sig2) == 1.0


def test_lsh_finds_exact_duplicate_candidate():
    mh = MinHasher(num_perm=64, seed=123)
    sig = mh.signature(preprocess("alpha beta gamma delta", shingle_size=2))
    idx = LSHIndex(num_bands=16)
    idx.add("a", sig)
    idx.add("b", sig)
    assert ("a", "b") in idx.candidate_pairs()


def test_simhash_identical_texts_have_zero_hamming_distance():
    h1, h2 = simhash_texts(["duplicate text example", "duplicate text example"])
    assert hamming_distance(h1, h2) == 0
    assert simhash_similarity(h1, h2) == 1.0



def test_prepare_pan11_synthetic_xml(tmp_path):
    from plagiarism_engine.pan11 import build_pan11_pairs

    root = tmp_path / "pan11"
    sus_dir = root / "suspicious-document"
    src_dir = root / "source-document"
    sus_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)

    sus_text = "Intro text. " + "alpha beta gamma delta epsilon zeta eta theta " * 4 + "Tail."
    src_text = "alpha beta gamma delta epsilon zeta eta theta " * 5
    neg_text = "finance market stock portfolio asset allocation risk return " * 5
    (sus_dir / "suspicious-document00001.txt").write_text(sus_text, encoding="utf-8")
    (src_dir / "source-document00001.txt").write_text(src_text, encoding="utf-8")
    (src_dir / "source-document00002.txt").write_text(neg_text, encoding="utf-8")

    offset = sus_text.index("alpha")
    xml = (
        f'<document reference="suspicious-document00001.txt">\n'
        f'  <feature name="plagiarism" this_offset="{offset}" this_length="120" '
        f'source_reference="source-document00001.txt" source_offset="0" source_length="120" />\n'
        f'</document>'
    )
    (sus_dir / "suspicious-document00001.xml").write_text(xml, encoding="utf-8")

    out = tmp_path / "pairs.csv"
    df = build_pan11_pairs(root, out, negatives_per_positive=1, min_chars=20)
    assert out.exists()
    assert set(df["label"].tolist()) == {0, 1}
    assert {"text_a", "text_b", "label", "suspicious_file", "source_file"}.issubset(df.columns)
