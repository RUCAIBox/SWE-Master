set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000


export WANDB_API_KEY=xx
export WANDB_MODE=online

# Find the directory where rllm package is located
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
echo ${RLLM_DIR}
echo "Python executable:"
python3 -c "import sys; print(sys.executable)"

echo "Python version:"
python3 -c "import sys; print(sys.version)"

echo "Pip version and location:"
python3 -m pip --version

echo "Installed packages:"
python3 -m pip list

echo "========"


export HYDRA_FULL_ERROR=1
export PYTHONPATH=./DeepSWE_RL/rllm/rllm:$PYTHONPATH


PROJECT_NAME="swe_agent_hls_rl_H800_32b_128K_4B"
EXPERIMENT_NAME="swe-agent-rl_docker_all_128K-timeout3600_4b"

TS=$(TZ="Asia/Shanghai" date +"%Y-%m-%d_%H-%M-%S")   

EXPERIMENT_NAME="${EXPERIMENT_NAME}_${TS}"

echo "Project: ${PROJECT_NAME}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Tensorboard dir: ./DeepSWE_RL/rllm/rllm/tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# echo "TENSORBOARD_DIR=$TENSORBOARD_DIR"

DEFAULT_LOCAL_DIR="./DeepSWE_RL/rllm/rllm/train_output/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="./DeepSWE_RL/rllm/rllm/train_output_rollout/${PROJECT_NAME}/${EXPERIMENT_NAME}"

mkdir -p "${DEFAULT_LOCAL_DIR}" "${ROLLOUT_DATA_DIR}"

echo "DEFAULT_LOCAL_DIR: ${DEFAULT_LOCAL_DIR}"
echo "ROLLOUT_DATA_DIR: ${ROLLOUT_DATA_DIR}"


python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=loop \
    data.train_files=/code/sunshuang/policy_rl/ss_debug/rllm/rllm/data/datasets/SWE_REBENCH_AND_GYM_OOD_RL_ADD_MAKE_TEST_SPEC/train_00000_verl.parquet \
    data.val_files=/code/sunshuang/policy_rl/ss_debug/rllm/rllm/data/datasets/SWE_BENCH_VERIFIED_TRY_DEBUG/test_00000_verl.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=512 \
    data.max_prompt_length=8192 \
    data.max_response_length=122880 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=/code/songhuatong/sft_models/qwen_3-4B-ins_80K_1206_minimax_glm_bon_filter_not_dedup_remove01 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16500 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.mask_truncated_samples=False \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='deepscaler-agent' \
    trainer.experiment_name='swe-agent-rl' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    env.name=swe_arsenal \
    +env.env_args.delete_image=False \
    +env.env_args.scaffold="openhands" \
    +env.env_args.backend="docker" \
    agent.name=sweagent \
    agent.max_steps=150 \
    agent.overlong_filter=True \
    agent.trajectory_timeout=3600 \
    agent.async_engine=True \
    +agent.agent_args.scaffold="openhands" \
    trainer.total_epochs=5 \