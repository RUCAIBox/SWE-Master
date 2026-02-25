export OPENAI_API_KEY='xx'

python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./results/glm-sweb-full-lsp" \
  --max_workers 20 \
  --start_idx 0 \
  --k 500 \
  --dataset "/code/songhuatong/swe_datasets/SWE-Bench-Verified_disk" \
  --split "test" \
  --llm_name 'openai/nbgexp' \
  --use_fn_calling True \
  --exp_name glm-sweb-full-lsp \
  --temperature 1.0 \
  --max_steps 150 \
  --max_steps_absolute 150 \
  --backend "docker" \
  --prepull_images False \
  --scaffold "openhands" \
  --use_lsp False \
  --used_yaml "./src/r2egym/agenthub/config/openhands/openhands_sp_fn_calling.yaml" \

