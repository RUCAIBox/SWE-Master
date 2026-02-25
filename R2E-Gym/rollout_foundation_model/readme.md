This README provides an overview of the environment variables and command-line arguments used in the execution script for the R2E-Gym Agent.

# Agent Execution Configuration Guide

## 1. Environment Variables
Before running the script, you must configure your API access:

| Variable | Description |
| :--- | :--- |
| `OPENAI_API_KEY` | Your authentication key for the LLM provider. |
| `OPENAI_API_BASE` | The base URL/endpoint for the API (e.g., a proxy or specific model provider). |

---

## 2. General Execution Parameters
These arguments control the scope and parallelism of the evaluation run.

| Parameter | Description |
| :--- | :--- |
| `--traj_dir` | The output directory where trajectory JSON files and execution logs will be saved. |
| `--max_workers` | Number of parallel worker threads/processes to run (set to 1 for serial debugging). |
| `--start_idx` | The starting index of the instance in the dataset to begin evaluation. |
| `--k` | Number of times to run each instance (useful for pass@k metrics). |
| `--dataset` | Path to the local dataset directory (e.g., SWE-Bench-Verified). |
| `--split` | The dataset split to use (e.g., `test`, `dev`, `train`). |
| `--exp_name` | A unique label for this experiment run to help organize results. |

---

## 3. LLM & Agent Logic Settings
These arguments define how the LLM behaves and which prompting scaffold is used.

| Parameter | Description |
| :--- | :--- |
| `--llm_name` | The model identifier (formatted for `litellm`, e.g., `openai/deepseek-ai/DeepSeek-V3.2`). |
| `--temperature` | Sampling temperature for the LLM. `1.0` allows for more variety; `0` is deterministic. |
| `--use_fn_calling` | Boolean. If `True`, uses the model's native tool-calling (function calling) capabilities. |
| `--scaffold` | The interaction framework to use (`openhands`, `r2egym`, or `sweagent`). Defines the prompt style and toolset. |
| `--used_yaml` | Path to the configuration YAML file containing system prompts and tool definitions. |
| `--max_steps` | The soft limit for steps. The agent is warned when it reaches this limit. |
| `--max_steps_absolute` | The hard limit for steps. The agent is forced to stop after this many turns. |

---

## 4. Environment & Runtime Settings
Settings related to the execution environment (Docker).

| Parameter | Description |
| :--- | :--- |
| `--backend` | The runtime environment for code execution (default: `docker`). |
| `--prepull_images` | If `True`, attempts to download all necessary Docker images before starting. |
| `--ip` | The IP address of the Docker host or registry. |
| `--use_lsp` | If `True`, enables Language Server Protocol for better code navigation/diagnostics. |

---

## 5. Memory & Compression Settings (Advanced)
These parameters control the "Long-term Memory" system, which summarizes old turns to fit into the LLM's context window.

| Parameter | Description |
| :--- | :--- |
| `--enable_compression` | Boolean. If `True`, enables the history summarization/compression system. |
| `--compression_trigger_step` | The step number that triggers the first compression event. |
| `--summary_window` | The number of previous turns included in a single summary block. |
| `--keep_recent` | The number of most recent turns kept as "full text" (no compression) to maintain immediate context. |
| `--use_single_turn_summary` | If `False`, uses aggregated summaries (multiple turns at once). If `True`, summarizes each turn individually. |
| `--memory_output_path` | Directory where compressed memory logs and summary reports are stored for analysis. |

---

## Example Usage
## normal inference
```bash 
bash /code/huanglisheng/work/R2E-Gym-opensource/rollout/run_deepseek_v3.sh
```

## inference with LSP Tool
```bash 
bash /code/huanglisheng/work/R2E-Gym-opensource/rollout/lsp/run_deepseek_v3_lsp.sh
```

## inference with memory 
```bash 
bash /code/huanglisheng/work/R2E-Gym-opensource/rollout/memory/run_deepseek_v3_memory.sh
```