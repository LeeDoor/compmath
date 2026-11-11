#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
REQUIREMENTS="$ROOT_DIR/requirements.txt"
REQUIREMENTS_HASH="$(sha256sum "$REQUIREMENTS" | cut -d ' ' -f1)"
HASH_FILE="$VENV_DIR/.requirements.sha256"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
fi

if [[ ! -f "$HASH_FILE" || "$(cat "$HASH_FILE")" != "$REQUIREMENTS_HASH" ]]; then
    "$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS"
    printf '%s\n' "$REQUIREMENTS_HASH" > "$HASH_FILE"
fi

exec "$VENV_DIR/bin/python" "$ROOT_DIR/main.py"
