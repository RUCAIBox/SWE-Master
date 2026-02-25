# Copyright under Agentica Project.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray

import os 

@ray.remote
def check_worker_env(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.system(f"pip list > {output_path}")
    return f"Worker env saved to {output_path}"

# Local application imports

@hydra.main(config_path="../config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo_agent(config)


def run_ppo_agent(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"pip": ["sentence_transformers",'polars',"swebench==3.0.2","dm-tree","gym>=0.26.2","kubernetes>=32.0.1","fire",
        "./swebench_fork_swegym-2.0.13-py3-none-any.whl",
        "./swebench_fork_swerebench-4.0.3-py3-none-any.whl",
        "./swesmith-0.0.7-py3-none-any.whl","simple-parsing>=0.1.6",
        "together>=1.3.5","markdown>=3.7","pexpect>=4.9.0","libtmux>=0.40.1","bashlex>=0.18","google-cloud-aiplatform>=1.77.0","litellm>=1.58.2",
        "seaborn>=0.13.2","orjson>=3.10.18","gpustat>=1.1.1"],
                "working_dir": "./DeepSWE_RL/rllm/verl",
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "PYTHONPATH": "./DeepSWE_RL/rllm", "WANDB_API_KEY": "xx"}})

    # output_log = "./rl_env_no_uv_new.txt"
    # print(ray.get(check_worker_env.remote(output_log)))

    ray.get(train_agent.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def train_agent(config, agent_class=None, env_class=None, agent_args=None, env_args=None):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf
    import sys
    
    # sys.path.insert(0, "/code/sunshuang/data_pipeline/R2E-Gym-fork_sim/src")
    sys.path.insert(0, "./R2E-Gym/src")
    # sys.path.insert(0, "/code/songhuatong/dockerllm_rl/rllm/verl")
    sys.path.insert(0, "./DeepSWE_RL/rllm/verl")

    print("===="*100)
    import sys
    print(sys.path)
    print("===="*100)
    import verl.trainer.ppo.reward as reward_moulde
    print(reward_moulde.__file__)
    print("===="*100)

    from verl.utils.fs import copy_local_path_from_hdfs
    from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
    from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
    from verl.trainer.ppo.reward import load_reward_manager
    
    
    # kill

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    # processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.critic.strategy in ["fsdp", "fsdp2"]
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    if env_class is None:
        print("===Check env.name=="*20)
        print(f"config.env.name:{config.env.name}")
        print("===Check ENV_CLASS_MAPPING=="*20)
        # print(f"ENV_CLASS_MAPPING:{ENV_CLASS_MAPPING}")
        
        import rllm
        import os
        print("===Check absolute path of rllm=="*20)
        print(os.path.dirname(os.path.dirname(rllm.__file__)))


        env_class = ENV_CLASS_MAPPING[config.env.name]
    
    if agent_class is None:
        agent_class = AGENT_CLASS_MAPPING[config.agent.name]

    env_args = env_args or {}
    agent_args = agent_args or {}
    if config.env.get("env_args") is not None:
        env_args.update(config.env.get("env_args"))
    if config.agent.get("agent_args") is not None:
        agent_args.update(config.agent.get("agent_args"))

    # print("===Check env_args=="*20)
    # print(f"env_args:{env_args}")

    # print("===Check agent_args=="*20)
    # print(f"agent_args:{agent_args}")
    

    trainer = AgentPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        env_class=env_class,
        agent_class=agent_class,
        env_args=env_args,
        agent_args=agent_args,
    )

    print(f"Initializing trainer...")
    trainer.init_workers()
    print(f"Initialized trainer done.")
    print(f"start training...")
    trainer.fit_agent()
    print(f"training done.")


if __name__ == "__main__":
    main()
