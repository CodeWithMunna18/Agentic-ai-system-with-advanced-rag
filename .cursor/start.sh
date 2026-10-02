#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"

wait_for_api() {
  for _ in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000/status >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "API did not become ready on port 8000" >&2
  return 1
}

if ! tmux has-session -t api 2>/dev/null; then
  tmux new-session -d -s api \
    "cd '$ROOT' && exec python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/api.log"
fi

wait_for_api

if ! tmux has-session -t client 2>/dev/null; then
  tmux new-session -d -s client \
    "cd '$ROOT/client' && exec npm run dev 2>&1 | tee /tmp/client.log"
fi

for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:5173 >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done

echo "Vite dev server did not become ready on port 5173" >&2
exit 1
