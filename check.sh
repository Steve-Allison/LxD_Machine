#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== format ==="
pixi run fmt

echo "=== lint ==="
pixi run lint

echo "=== typecheck ==="
pixi run typecheck

echo "=== test ==="
pixi run test
