# Semantic Plagiarism Engine

A command-line plagiarism and near-duplicate detection engine for the third Data Mining project. The implementation compares two algorithmic paths:

1. **Word Shingling + MinHash + LSH**
2. **TF-IDF Weighted SimHash**

The code is intentionally modular and avoids ready-made implementations of MinHash, LSH, or SimHash. `numpy` and `pandas` are used only for array and tabular processing.

---

## Project Structure

```text
semantic-plagiarism-engine/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .github/workflows/tests.yml
├── docs/
│   ├── project_spec.tex
│   ├── project_spec.pdf
│   ├── project_report.tex
│   ├── project_report.pdf
│   └── PAN11_DATASET.md
├── data/
│   ├── sample_corpus/
│   ├── raw/          # ignored by git for large datasets
│   └── processed/    # ignored by git for generated datasets
├── src/plagiarism_engine/
│   ├── preprocessing.py
│   ├── minhash.py
│   ├── lsh.py
│   ├── simhash.py
│   ├── dataset.py
│   ├── pan11.py
│   ├── evaluation.py
│   └── cli.py
├── notebooks/
│   └── exploration.ipynb
├── scripts/
│   └── download_pan11.sh
├── tests/
│   └── test_engine.py
└── outputs/
    ├── two_file_compare.json
    ├── candidates.csv
    ├── metrics.csv
    └── pair_predictions.csv
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt -e .
```

---

## CLI Commands

### 1. Compare two files

```bash
python -m plagiarism_engine.cli compare \
  --file-a data/sample_corpus/doc_01.txt \
  --file-b data/sample_corpus/doc_02.txt \
  --output outputs/two_file_compare.json
```

The JSON output contains exact Jaccard, estimated MinHash similarity, SimHash similarity, Hamming distance, and shingle counts.

### 2. Search a corpus folder

```bash
python -m plagiarism_engine.cli corpus \
  --data data/sample_corpus \
  --threshold 0.25 \
  --shingle-size 3 \
  --output outputs/candidates.csv
```

The corpus command first builds MinHash signatures, indexes them with LSH bands, and only compares candidate pairs sharing at least one LSH bucket. The output includes exact Jaccard, MinHash similarity, SimHash similarity, and threshold flags.

### 3. Evaluate labelled pairs

```bash
python -m plagiarism_engine.cli pairs \
  --pairs data/raw/train.csv \
  --text-col-a question1 \
  --text-col-b question2 \
  --label-col is_duplicate \
  --limit 5000 \
  --output outputs/metrics.csv \
  --predictions-output outputs/pair_predictions.csv
```

The metrics file reports accuracy, precision, recall, F1-score, confusion counts, and average runtime per pair.



### Download PAN-PC-11 from Zenodo

The official dataset linked in the project brief is **PAN Plagiarism Corpus 2011 (PAN-PC-11)**, DOI `10.5281/zenodo.3250095`. It is distributed as a two-part RAR archive, so the raw files are not committed to this repository. To download them locally under `data/raw/`:

```bash
bash scripts/download_pan11.sh
```

After downloading, extract `pan-plagiarism-corpus-2011.part1.rar` with `unrar` or `7z`; the second part is used automatically by the extractor. Then run `prepare-pan11` as shown below. More details are in `docs/PAN11_DATASET.md`.

### 4. Prepare PAN-PC-11 labelled pairs

After downloading and extracting both PAN-PC-11 archive parts locally, keep the raw files under `data/raw/` and do not commit them to GitHub.

Expected local placement example:

```text
semantic-plagiarism-engine/
└── data/raw/pan11/
    └── external-detection-corpus/
        ├── suspicious-document*/
        ├── source-document*/
        └── *.xml / *.txt files
```

Convert PAN XML annotations into a labelled pair CSV:

```bash
python -m plagiarism_engine.cli prepare-pan11 \
  --pan-root data/raw/pan11 \
  --output data/processed/pan11_pairs.csv \
  --max-positive 5000 \
  --negatives-per-positive 1 \
  --segment-mode annotated
```

Then evaluate both algorithms on the generated PAN pair file:

```bash
python -m plagiarism_engine.cli pairs \
  --pairs data/processed/pan11_pairs.csv \
  --text-col-a text_a \
  --text-col-b text_b \
  --label-col label \
  --limit 5000 \
  --output outputs/metrics_pan11.csv \
  --predictions-output outputs/pair_predictions_pan11.csv
```

You can also run candidate discovery on a folder of PAN documents:

```bash
python -m plagiarism_engine.cli corpus \
  --data data/raw/pan11/external-detection-corpus \
  --threshold 0.25 \
  --shingle-size 3 \
  --output outputs/candidates_pan11.csv
```

For the full PAN corpus, start with a small `--max-positive` value or a smaller copied subset of documents, then scale up after verifying the pipeline.

---

## Running Tests

```bash
pytest -q
```

GitHub Actions is configured in `.github/workflows/tests.yml` to run the same test command on pushes and pull requests.

---

## Method Summary

### Preprocessing

Each document is normalized with Unicode NFKC, lowercasing, punctuation removal, Persian/Arabic character unification, stopword removal, tokenization, and word shingling. The default shingle size is 3 words. Empty or very short documents are handled safely.

### Exact Jaccard Similarity

For shingle sets `A` and `B`:

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

Directly computing this for all document pairs costs `O(n^2)` pair comparisons, so it becomes expensive for large corpora.

### MinHash

Each shingle is converted to a stable 64-bit hash. A family of deterministic universal hash functions generates a signature vector. The similarity of two documents is estimated by the fraction of equal signature positions.

### LSH

Each MinHash signature is split into bands. Documents sharing a bucket in at least one band become candidate pairs. This reduces the number of final comparisons compared with all-to-all matching.

### TF-IDF Weighted SimHash

For each document, token TF-IDF weights are accumulated in a 64-dimensional vector according to token hash bits. The final SimHash bit is 1 when the corresponding vector dimension is non-negative, otherwise 0. Similar documents are expected to have a small Hamming distance.

---

## PAN-PC-11 Dataset Workflow

The `prepare-pan11` command parses PAN XML annotations, builds positive pairs from annotated suspicious/source passages, samples negative pairs from unrelated source documents, and writes a reproducible CSV under `data/processed/`. The raw PAN archive and extracted corpus must stay under `data/raw/`, which is ignored by git.

## Notes About Large Datasets

The project can be run on PAN-PC-11, Stack Exchange duplicate pairs, or Quora Question Pairs after downloading the data locally. Large raw data should remain under `data/raw/`, which is ignored by git. Only small examples and reproducible outputs should be committed. The small sample `data/raw/train.csv` and `data/processed/sample_pairs.csv` are explicitly allowed for reproducible classroom tests. For PAN-PC-11, commit only scripts, README instructions, reports, and generated summary outputs such as `outputs/metrics_pan11.csv`; do not commit the raw `.rar` files or extracted corpus.

