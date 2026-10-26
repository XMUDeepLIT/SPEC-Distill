#!/bin/bash
# Link expert rollout files from a source directory to a target directory.
# Usage: bash init_expert_files.sh <SRC_DIR> <TARGET_DIR>

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: bash $0 <SRC_DIR> <TARGET_DIR>"
  exit 1
fi

SRC="$1"
TARGET="$2"

if [ ! -d "$TARGET" ]; then
  echo "Target directory does not exist; creating: $TARGET"
  mkdir -p "$TARGET"
fi

cd "$TARGET" || { echo "Failed to enter target directory: $TARGET"; exit 1; }

echo "Current directory: $(pwd)"
echo "Creating symlinks from $SRC ..."

FILES=(
  "-1.jsonl" "-1.pt"
  "-2.jsonl" "-2.pt"
  "-3.jsonl" "-3.pt"
  "-4.jsonl" "-4.pt"
  "-5.jsonl" "-5.pt"
  "0.jsonl"  "0.pt"
)

for f in "${FILES[@]}"; do
  src="$SRC/$f"
  if [ -f "$src" ]; then
    ln -sf "$src" .
    echo "Linked: $f"
  else
    echo "Missing file: $src"
  fi
done

echo "Done."
