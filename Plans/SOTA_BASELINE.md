# SOTA Baseline — 2026-07-20

Frozen before the SOTA Full Vision implementation so later phases can prove lift.

## Retrieval eval (`pixi run retrieval-check`)

| Metric | Value |
| --- | --- |
| Config | `config.yaml` (embed `text-embedding-3-small`, data `data/openai`) |
| Eval set | `tests/eval/eval_set.json` (20 core questions at freeze) |
| Mean Recall@10 | **1.000** |
| Mean MRR@10 | **0.713** |
| Notes | Reranker warnings present (Ollama/`qwen3-reranker` unreachable); dense+fusion path still scored. Run id tag: `sota-baseline-2026-07-20`. |

## Quality eval (`pixi run eval-quality`)

Latest historical run in `data/openai/eval_quality_runs.jsonl` (2026-06-25, judge `gpt-4o-mini`, 5 golden questions):

| Metric | Value |
| --- | --- |
| Mean faithfulness | 0.982 |
| Mean answer relevance | 0.695 |
| Mean context precision | 0.826 |
| Composite | 0.818 |

Re-run quality eval after Phase 1 gate; do not treat the June run as identical hardware/corpus.

## Design-oriented subset

From 2026-07-20 the eval set includes a `category: "design"` subset (≥15 cases) for jargon / instructional-design queries (Bloom application, modality choice, cognitive load, objectives, transfer, microlearning, scenario design, Adobe-style field enablement framing). Track subset Recall@10 / MRR@10 separately when reporting Phase gates.
