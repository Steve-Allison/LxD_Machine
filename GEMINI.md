# GEMINI.md - LxD_Machine

## 🔗 Inheritance
- Inherits from: `/Users/steveallison/.gemini/gemini.md`

## 🎯 SOTA Mandate
- **NO ASSUMPTIONS**: Verify all paths, names, and states before acting.
- **PRODUCTION-READY CODE**: No stubs, no placeholders, complete implementations only.
- **INTEGRITY PROTOCOL**: Maintain system and architectural integrity at all times.
- **VERIFICATION IS FINALITY**: Nothing is complete until verified by tests or shell commands.

## 🛠 Tech Stack
- **Environment & Package Manager**: Pixi (`pixi.toml`, conda-forge + PyPI, osx-arm64)
- **Runtime**: Python 3.14 (free-threaded ready, modern standard library)
- **Stores**: LanceDB (canonical vector store) + SQLite (relational metadata, graph, runs)
- **Local Acceleration**: MLX & MLX-LM (Apple Silicon native embeddings & reranking)
- **API & LLM Integration**: OpenAI API, FastMCP (>=3.0) stdio server
- **Ingestion & Ontology**: Docling 2.0, LinkML / Central_Configs ontologies, pyahocorasick
- **Code Quality**: Ruff (target py314), Pyright (strict mode), Pytest (asyncio, benchmarks)

