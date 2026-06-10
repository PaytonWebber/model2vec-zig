#!/usr/bin/env bash
# Fetch a model2vec model from HuggingFace into models/<name> for the parity
# test. Defaults to potion-base-8M (~30 MB).
set -euo pipefail

NAME="${1:-potion-base-8M}"
ORG="${2:-minishlab}"
DEST="models/$NAME"

mkdir -p "$DEST"
for f in config.json tokenizer.json model.safetensors; do
    if [[ ! -f "$DEST/$f" ]]; then
        echo "fetching $f"
        curl -fsSL "https://huggingface.co/$ORG/$NAME/resolve/main/$f" -o "$DEST/$f"
    fi
done
echo "model ready in $DEST"
