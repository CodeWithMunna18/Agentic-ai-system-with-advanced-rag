#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Cloud Agent secrets override any committed .env value.
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
  printf 'GEMINI_API_KEY=%s\n' "$GEMINI_API_KEY" > .env
fi

python3 -m pip install --user -r requirements.txt

if [[ ! -f client/.env ]]; then
  cp client/.env.example client/.env
fi

cd client
npm ci
