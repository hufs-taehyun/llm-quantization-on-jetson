# llm-quantization-on-jetson

Small experiments on what weight quantization costs and buys for small LLMs on a Jetson Orin Nano 8GB.
Each experiment fixes its question and pass criteria in `PLAN.md` before running, and reports results and limits in `README.md`.
The detailed documents are written in Korean.

| Experiment | What it does | Main result |
|---|---|---|
| [02_llm_quant_confidence](02_llm_quant_confidence/) | INT8 / INT4 round-to-nearest weight quantization written by hand in PyTorch (per-channel W8, group-wise W4). Two models (Qwen2.5-1.5B, SmolLM2-1.7B) x two datasets (ARC-Challenge, MMLU), scored from next-token answer probabilities. | W8 is indistinguishable from FP16 in all four model-dataset pairs. W4 raises confidently wrong answers on Qwen (+3.8 pp on ARC, +2.7 pp on MMLU) but lowers them on SmolLM2, so the conclusion is limited to Qwen. |
| [04_llm_gguf_on_device](04_llm_gguf_on_device/) | The same models as real quantized files (GGUF F16 / Q8_0 / Q4_K_M) run with llama.cpp on the device: speed, memory, power, and accuracy measured on the same files. | 4-bit generates about 2x faster than 16-bit (Qwen 17.4 → 39.9 tok/s, SmolLM2 20.0 → 38.7 tok/s) at under half the energy per token. Accuracy loss is 2.9 pp on Qwen and 8.5 pp on SmolLM2. 8-bit loses at most 1.0 pp. |

## How the comparisons are made

- Criteria first: the judgment rules (R1–R3 in 02, D1–D3 in 04) are committed before the result commits.
- Differences are reported with question-level bootstrap confidence intervals (10,000 resamples, seed 0) and Holm correction.
- Analyses added after seeing results are marked as post hoc in each README.
- Raw per-question predictions and benchmark logs are in each experiment's result folders. Datasets and model files are not included.

## Environment

Jetson Orin Nano 8GB, JetPack 6.2.1, 25 W mode. PyTorch 2.8.0, Transformers 4.57.6, llama.cpp (commit `ff6b8d1`, CUDA build).

## Limits

- Two models of similar size and multiple-choice benchmarks only; generation quality is not evaluated.
- The two 4-bit schemes differ (hand-written round-to-nearest in 02, Q4_K_M in 04), so their numbers are not directly comparable.
- Power is board input power from `tegrastats`; energy per token is an approximation.
