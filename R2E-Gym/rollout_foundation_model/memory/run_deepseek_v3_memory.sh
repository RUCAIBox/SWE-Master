export OPENAI_API_KEY='xx'

uv run python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./results/deepseek-v3-sweb-500" \
  --max_workers 1 \
  --start_idx 0 \
  --k 1 \
  --dataset "/code/songhuatong/swe_datasets/SWE-Bench-Verified_disk" \
  --split "test" \
  --llm_name 'openai/deepseek-ai/DeepSeek-V3.2' \
  --use_fn_calling True \
  --exp_name debug-memory \
  --temperature 1.0 \
  --max_steps 20 \
  --max_steps_absolute 20 \
  --backend "docker" \
  --prepull_images False \
  --scaffold "openhands" \
  --ip "10.106.35.101" \
  --use_lsp False \
  --used_yaml "./src/r2egym/agenthub/config/openhands/openhands_sp_fn_calling_memory.yaml" \
  --enable_compression True \
  --summary_window 10 \
  --keep_recent 5 \
  --compression_trigger_step 15 \
  --use_single_turn_summary False \
  --memory_output_path "/code/huanglisheng/work/R2E-Gym-plan-with-files/memory_cache/deepseek-memory-agg/SWE-Bench-Verified_random_50" \
