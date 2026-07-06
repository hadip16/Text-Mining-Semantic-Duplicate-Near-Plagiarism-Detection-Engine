# PAN-PC-11 Dataset Notes

The requested dataset is **PAN Plagiarism Corpus 2011 (PAN-PC-11)**, DOI `10.5281/zenodo.3250095`.

Zenodo files:

| file | size | md5 |
|---|---:|---|
| `pan-plagiarism-corpus-2011.part1.rar` | 1.0 GB | `b2930f859497dd48ba5bb606d3f4a4f3` |
| `pan-plagiarism-corpus-2011.part2.rar` | 703.9 MB | `b23d86c17a47d2bfbdc4c314ea5810df` |

Raw PAN files are intentionally not included in the GitHub repository because the project guide requires large raw data to remain outside git. Use:

```bash
bash scripts/download_pan11.sh
```

Then extract the archive and run:

```bash
python -m plagiarism_engine.cli prepare-pan11 \
  --pan-root data/raw/pan11 \
  --output data/processed/pan11_pairs.csv \
  --max-positive 5000 \
  --negatives-per-positive 1 \
  --segment-mode annotated

python -m plagiarism_engine.cli pairs \
  --pairs data/processed/pan11_pairs.csv \
  --text-col-a text_a \
  --text-col-b text_b \
  --label-col label \
  --limit 5000 \
  --output outputs/metrics_pan11.csv \
  --predictions-output outputs/pair_predictions_pan11.csv
```

The `prepare-pan11` command parses XML plagiarism annotations, extracts suspicious/source passages, and samples deterministic negative pairs.
