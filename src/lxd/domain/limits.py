"""Shared hard limits used across retrieval, config, and MCP surfaces."""

from typing import Final

# Maximum dense/hybrid candidate count accepted by the query pipeline.
# Adaptive-router breadth knobs and MCP ``limit`` fields must share this
# ceiling so a legal config/API value can never raise ``ValueError`` at
# query time.
MAX_RETRIEVAL_LIMIT: Final[int] = 50
