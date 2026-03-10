#!/usr/bin/env bash
set -e
cd /Users/steveallison/AI_Projects+Code/LxD_Machine
PIXI_BIN=.pixi/envs/default/bin

echo "=== ruff fix ==="
$PIXI_BIN/ruff check --fix src tests || true

echo "=== lint ==="
$PIXI_BIN/ruff check src tests && echo "LINT OK" || echo "LINT FAILED"

echo "=== typecheck ==="
$PIXI_BIN/pyright src 2>&1 | tail -5

echo "=== test ==="
$PIXI_BIN/pytest tests/ -v --tb=short 2>&1
