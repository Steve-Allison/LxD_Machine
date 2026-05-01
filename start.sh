#!/usr/bin/env bash
# LxD Machine — interactive launcher
# Lets you pick from the available pixi tasks, with a default
# "ingest + build-graph" flow for handling new corpus files.

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v pixi >/dev/null 2>&1; then
  echo "Error: pixi is not installed or not on PATH." >&2
  echo "Install via: brew install pixi" >&2
  exit 1
fi

cat <<'BANNER'
╔══════════════════════════════════════════════════════════════╗
║                    LxD Machine — Launcher                    ║
╚══════════════════════════════════════════════════════════════╝
BANNER

echo
echo "Pick an action:"
echo
echo "  Preflight (free, fast)"
echo "   0) Preflight checks             (verify schema + stores before spending)"
echo
echo "  Pipeline"
echo "   1) Preflight + Ingest + Build-Graph (recommended for new files)"
echo "   2) Ingest only                  (incremental)"
echo "   3) Ingest --full                (full rebuild — uses cache, no extra spend if model unchanged)"
echo "   4) Build-Graph only             (incremental, resumable)"
echo "   5) Build-Graph --full           (re-extract claims; will prompt to confirm)"
echo
echo "  Status / Inspection"
echo "   6) Corpus status"
echo "   7) Graph status"
echo "   8) Batch status"
echo
echo "  Services"
echo "   9) Launch MCP server"
echo "  10) Run evaluation suite"
echo
echo "  Dev"
echo "  11) Tests        (pixi run test)"
echo "  12) Lint         (pixi run lint)"
echo "  13) Format       (pixi run fmt)"
echo "  14) Typecheck    (pixi run typecheck)"
echo "  15) Collect batch"
echo
echo "   q) Quit"
echo

read -rp "Selection [1]: " choice
choice="${choice:-1}"

run() {
  echo
  echo "▶ pixi run $*"
  echo "─────────────────────────────────────────────────────────────"
  pixi run "$@"
}

case "$choice" in
  0)
    run preflight
    ;;
  1)
    run preflight
    echo
    run ingest
    echo
    run build-graph
    echo
    echo "✓ Preflight + Ingest + Build-Graph complete."
    ;;
  2)  run ingest ;;
  3)
    read -rp "Full rebuild re-embeds everything (slow, costs API calls). Continue? [y/N]: " ok
    [[ "${ok,,}" == "y" ]] || { echo "Aborted."; exit 0; }
    run ingest --full
    ;;
  4)  run build-graph ;;
  5)  run build-graph --full ;;
  6)  run status ;;
  7)  run graph-status ;;
  8)  run batch-status ;;
  9)  run mcp ;;
  10) run eval ;;
  11) run test ;;
  12) run lint ;;
  13) run fmt ;;
  14) run typecheck ;;
  15) run collect-batch ;;
  q|Q) echo "Bye."; exit 0 ;;
  *)
    echo "Unknown selection: $choice" >&2
    exit 1
    ;;
esac
