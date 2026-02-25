export OPENAI_API_KEY='xx'
export OPENAI_API_BASE='http://xx.xxx.xxx.xxx:8000/v1'
cd ./R2E-Gym
python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./results/swe-master-eval" \
  --max_workers 20 \
  --start_idx 0 \
  --k 500 \
  --dataset "/code/songhuatong/swe_datasets/SWE-Bench-Verified_disk" \
  --split "test" \
  --llm_name 'hosted_vllm/swe-master-sft' \
  --use_fn_calling False \
  --exp_name swe-master-eval \
  --temperature 0.7 \
  --max_steps 150 \
  --max_steps_absolute 150 \
  --backend "docker" \
  --prepull_images False \
  --scaffold "openhands" \
  --ip "xxx" \
  --used_yaml "./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml"
