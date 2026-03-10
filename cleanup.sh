#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${SCRIPT_DIR}/pixi.toml" || ! -d "${SCRIPT_DIR}/Plans" ]]; then
  echo "cleanup.sh must be run from the LxD_Machine project root." >&2
  exit 1
fi

DRY_RUN=0
INCLUDE_ENV=0

usage() {
  cat <<'EOF'
Usage: ./cleanup.sh [--dry-run] [--include-env]

Removes generated runtime/build/cache artifacts for a fresh project run.

Default cleanup removes:
- data store contents under data/
- project runtime logs and pid files
- Python bytecode and __pycache__ directories
- Ruff/Pytest/Pyright/Mypy caches
- coverage outputs
- build/dist/egg artifacts
- .firecrawl scratch data
- .DS_Store files inside this repo

Optional:
- --dry-run     Show what would be removed without changing anything
- --include-env Also remove .pixi/ from this repo
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    --include-env)
      INCLUDE_ENV=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

run_cmd() {
  if (( DRY_RUN )); then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

remove_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    return 0
  fi
  run_cmd rm -rf "$path"
}

remove_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return 0
  fi
  run_cmd rm -f "$path"
}

stop_runtime_processes() {
  local runtime_dir="${SCRIPT_DIR}/data/runtime"
  local runtime_file

  [[ -d "$runtime_dir" ]] || return 0

  while IFS= read -r -d '' runtime_file; do
    local pid=""
    pid="$(python3 - <<'PY' "$runtime_file"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

pid = payload.get("pid")
print(pid if isinstance(pid, int) else "")
PY
)"

    if [[ -z "$pid" ]]; then
      continue
    fi

    if kill -0 "$pid" 2>/dev/null; then
      run_cmd kill -TERM "$pid"
    fi
  done < <(find "$runtime_dir" -type f -name '*.json' -print0)
}

remove_find_matches() {
  local search_root="$1"
  shift
  local args=("$@")
  local prune_args=()

  [[ -e "$search_root" ]] || return 0

  if (( INCLUDE_ENV )); then
    prune_args=( \( -path "${SCRIPT_DIR}/.git" \) -prune -o )
  else
    prune_args=( \( -path "${SCRIPT_DIR}/.git" -o -path "${SCRIPT_DIR}/.pixi" \) -prune -o )
  fi

  while IFS= read -r -d '' match; do
    if [[ -d "$match" ]]; then
      run_cmd rm -rf "$match"
    else
      run_cmd rm -f "$match"
    fi
  done < <(find "$search_root" "${prune_args[@]}" "${args[@]}" -print0)
}

echo "Cleaning generated project state under: ${SCRIPT_DIR}"

stop_runtime_processes

remove_path "${SCRIPT_DIR}/data"
run_cmd mkdir -p "${SCRIPT_DIR}/data"

remove_path "${SCRIPT_DIR}/.pytest_cache"
remove_path "${SCRIPT_DIR}/.ruff_cache"
remove_path "${SCRIPT_DIR}/.pyright"
remove_path "${SCRIPT_DIR}/.mypy_cache"
remove_path "${SCRIPT_DIR}/htmlcov"
remove_path "${SCRIPT_DIR}/dist"
remove_path "${SCRIPT_DIR}/build"
remove_path "${SCRIPT_DIR}/.eggs"
remove_path "${SCRIPT_DIR}/.firecrawl"
remove_file "${SCRIPT_DIR}/.coverage"
remove_file "${SCRIPT_DIR}/coverage.xml"
remove_file "${SCRIPT_DIR}/pyrightconfig.json"

if (( INCLUDE_ENV )); then
  remove_path "${SCRIPT_DIR}/.pixi"
fi

remove_find_matches "${SCRIPT_DIR}" -type d \( -name '__pycache__' -o -name '*.egg-info' \)
remove_find_matches "${SCRIPT_DIR}" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pyd' -o -name '.DS_Store' \)

echo "Cleanup complete."
