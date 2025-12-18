# VisionMetricAgent

VisionMetricAgent is a LangGraph-powered assistant that helps evaluate image generation experiments. It wraps classic perceptual metrics (PSNR, SSIM, LPIPS, CW-SSIM, FSIM, SAM, RMSE) and exposes them through an interactive CLI agent that can remember evaluation pairs, run batch scoring jobs, and export consolidated reports.

## Project layout

- `src/agent_cli.py` – CLI entry point that builds a ReAct-style agent with LangGraph. The agent keeps conversational memory, accepts user prompts, and routes work to the evaluation tools.
- `src/tools_eval.py` – Tool definitions the agent can call to manage evaluation runs: add directory pairs, list them, run metrics, and save the latest report.
- `src/session_state.py` – In-memory session management for folder pairs, pending requests, and the latest evaluation results.
- `src/metrics_backend.py` – Backend helpers that call the metric implementations, aggregate statistics, and render a multi-pair TXT report.
- `src/psnr_ssim_all.py` – Metric runner that pairs images by filename, resizes them, computes LPIPS, CW-SSIM, SSIM, FSIM, RMSE, PSNR, and SAM, and logs warnings for missing or unmatched files.
- `metric/`, `lpips/`, `image_similarity_measures/` – Metric implementations and third-party helpers used by the backend.
- `requirements.txt` – Python dependencies for the agent and metric stack.
- `test.py` – Minimal OpenAI/DashScope streaming example used to validate model connectivity.

## Getting started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure model access**
   The CLI uses `ChatOpenAI` with the DashScope-compatible endpoint. Set environment variables instead of hard-coding secrets:
   ```bash
   export OPENAI_API_KEY="<your-dashscope-key>"
   export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
   ```
   Then adjust `src/agent_cli.py` to read from the environment or pass your own configuration when constructing the LLM client.

3. **Prepare evaluation data**
   Organize ground-truth and generated images in separate folders with matching filenames. The metric runner will align files by name and resize them (default 256×256) before scoring.

4. **Run the agent CLI**
   ```bash
   python src/agent_cli.py
   ```
   Example conversation:
   - Add a pair: `添加目录对，名字是 demo，gt 是 /path/to/gt，gen 是 /path/to/gen`.
   - Run metrics: `计算 demo 的 psnr、ssim、lpips，分辨率 256`.
   - Save a report: `把上次结果保存到 reports/demo.txt`.

## Evaluation flow

1. **Add pairs** – `add_pair` validates directories and stores them in the session. If a pending evaluation exists, it can automatically continue and compute metrics immediately.
2. **Evaluate pairs** – `eval_pairs` checks metric names, selects requested pairs (or all), calls `evaluate_dirs`, and caches results for later export. If no pairs exist yet, it records the pending request so the agent can immediately continue once a pair is added.
3. **Compute metrics** – `evaluate_dirs` calls `SSIMs_PSNRs` to produce per-image arrays, then summarises mean/variance/min/max/std for each metric.
4. **Save reports** – `save_last_report` converts the latest multi-pair results into a TXT summary via `save_multi_report_txt`.

## Key behaviors and defaults

- Metrics supported: PSNR, SSIM, RMSE, LPIPS, CW-SSIM, FSIM, SAM.
- Image pairing: files are matched by basename; unmatched files trigger warnings but do not halt execution.
- Resizing: images are resized to the requested square resolution (default 256) before scoring.
- Hardware: LPIPS loads an AlexNet backbone and moves it to GPU when available; otherwise falls back to CPU.
- Logging & history: each evaluation is cached in-memory and appended to `logs/eval_history.jsonl` for recall across turns. Use the `list_eval_history` tool to inspect recent runs.

## Tips for extending or learning the codebase

- **Trace the agent loop** in `src/agent_cli.py` to see how LangGraph invokes the tool layer and preserves conversation state across turns.
- **Inspect `src/tools_eval.py`** to understand argument validation, pending-evaluation continuation, and how results are cached for reporting.
- **Review `src/psnr_ssim_all.py`** for details on how each metric is computed, device handling for LPIPS, and file-matching rules.
- **Add tests** around `metrics_backend.evaluate_dirs` using small synthetic image sets to validate statistics and edge cases (missing files, NaNs).
- **Parameterize secrets** by reading API keys and model endpoints from environment variables to keep credentials out of source control.
- **Consider batching** LPIPS computations on GPU if you process large datasets, and profile I/O versus compute to tune throughput.

## Troubleshooting

- Empty or mismatched folders will return empty metric arrays; check warnings in the log output.
- If LPIPS fails on a CPU-only environment, set `device="cpu"` in `SSIMs_PSNRs` or guard GPU usage before calling `.cuda()`.
- Ensure image filenames align between `gt_dir` and `gen_dir`; otherwise metrics will skip non-overlapping files.
