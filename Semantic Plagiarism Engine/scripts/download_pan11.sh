#!/usr/bin/env bash
# Download PAN-PC-11 from Zenodo into data/raw without committing the raw corpus.
# Usage: bash scripts/download_pan11.sh [target_directory]
set -euo pipefail

TARGET_DIR="${1:-data/raw/pan11_downloads}"
mkdir -p "$TARGET_DIR"

PART1="$TARGET_DIR/pan-plagiarism-corpus-2011.part1.rar"
PART2="$TARGET_DIR/pan-plagiarism-corpus-2011.part2.rar"

URL1="https://zenodo.org/records/3250095/files/pan-plagiarism-corpus-2011.part1.rar?download=1"
URL2="https://zenodo.org/records/3250095/files/pan-plagiarism-corpus-2011.part2.rar?download=1"

echo "Downloading PAN-PC-11 part 1 to $PART1"
curl -L -C - --retry 5 --retry-delay 10 -o "$PART1" "$URL1"

echo "Downloading PAN-PC-11 part 2 to $PART2"
curl -L -C - --retry 5 --retry-delay 10 -o "$PART2" "$URL2"

if command -v md5sum >/dev/null 2>&1; then
  (cd "$TARGET_DIR" && md5sum -c <<'MD5'
b2930f859497dd48ba5bb606d3f4a4f3  pan-plagiarism-corpus-2011.part1.rar
b23d86c17a47d2bfbdc4c314ea5810df  pan-plagiarism-corpus-2011.part2.rar
MD5
  )
else
  echo "md5sum not found; verify checksums manually."
fi

cat <<MSG

Download complete.
Extract the multi-part archive with one of the following commands:

  unrar x "$PART1" data/raw/pan11/

or, if 7-Zip is installed:

  7z x "$PART1" -odata/raw/pan11/

After extraction, create labelled pairs with:

  python -m plagiarism_engine.cli prepare-pan11 \\
    --pan-root data/raw/pan11 \\
    --output data/processed/pan11_pairs.csv \\
    --max-positive 5000 \\
    --negatives-per-positive 1 \\
    --segment-mode annotated
MSG
