# Scientific judge calibration

A controlled audit of factual correctness, response order and rhetorical style in a small LLM judge and a separately trained scalar reward model. Original synthetic tasks only; no private reviewer tasks or human-preference claims.

This project extends the [post-training lab](https://github.com/Mioulin/scientific-reasoning-post-training-lab). The companion lab must contain `scireason.toolstudy` (branch `research/verified-computation`) and its original SFT and reward checkpoints. Weights are not committed here. The exact dependency commit is recorded in `results/protocol.json` during execution.

## Protocol

64 problems across eight families. Five conditions: neutral; verbose incorrect/correct answer; confident incorrect/correct answer. Both A/B orders are evaluated. The 640 repeated decisions per policy are clustered into 64 problems, not treated as 640 independent observations. Three policies: base Qwen2.5-0.5B-Instruct, its earlier SFT adapter, and the earlier scalar reward adapter.

The first four instances per family form the 32-problem calibration split; the remaining four form the 32-problem calibration test split. Positive temperature is selected on neutral calibration data only; report raw and scaled Brier/NLL on the disjoint test under each intervention. This does not change pairwise classifications. A/B probabilities are conditional next-token choices. Scalar-reward sigmoid probabilities are not inherently calibrated.

## Reproduce

From the companion lab, install its pinned environment and generate/train its original SFT and reward models following its README. Prepare the fresh exclusion set with `PYTHONPATH=src .venv/bin/python -m scireason.toolstudy prepare`. Then, from this repository:

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check .
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ../scientific-reasoning-post-training-lab/.venv/bin/python study.py \
  --lab-root ../scientific-reasoning-post-training-lab --output results
```

`uv.lock` here pins the lightweight test/analysis environment; inference deliberately uses the companion lab's full pinned Transformers/PEFT environment. No paid APIs are used. Model cache must already exist. Reproduction in a new output directory preserves earlier results.

## Artefacts

`results/` holds problem provenance, raw probability/margin records, timings, clustered intervals, style sensitivity, reliability bins, held-out temperature calibration, report and comparison figure. `docs/learning_guide_ru.md` explains the design. Status is recorded in `PROGRESS.md`.

The approach is informed by [Zheng et al., Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685), which studies position and verbosity biases. This is a small controlled extension, not a reproduction of their human-agreement study. Base/SFT are one model family. Expert validation and cross-family replication are NOT RUN.

Original code: MIT. Original generated problems: CC0-1.0. Qwen base: Apache-2.0; the companion adapters inherit model requirements. This repository contains no model weights.
