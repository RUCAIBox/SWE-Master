export OPENAI_API_KEY='xx'
cd ./R2E-Gym
python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./results/ood/minimax-ood-rollout-good-swerebench-filter" \
  --max_workers 20 \
  --start_idx 0 \
  --k 1890 \
  --dataset "./my_datasets/swe-rebench-filtered" \
  --split "test" \
  --llm_name 'openai/models/arsenal-inf/MiniMax-M2' \
  --use_fn_calling True \
  --exp_name minimax-ood-rollout-good-swerebench-filter \
  --temperature 1.0 \
  --max_steps 150 \
  --max_steps_absolute 150 \
  --backend "docker" \
  --prepull_images False \
  --scaffold "openhands" \
  --use_lsp True \
  --used_yaml "./src/r2egym/agenthub/config/openhands/openhands_lsp_tools.yaml" \
  --use_demo "./rollout_foundation_model/lsp/swe_LSP_few_shot/few_shot_minimax.txt"
