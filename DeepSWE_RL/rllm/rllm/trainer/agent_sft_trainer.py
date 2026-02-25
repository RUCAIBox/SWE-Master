from torch.distributed.device_mesh import init_device_mesh

from rllm.agents.agent import Trajectory
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, create_sft_dataset
from verl.utils import hf_tokenizer
from verl.utils.device import get_device_name
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local


class AgentSFTTrainer(FSDPSFTTrainer):
    def __init__(self, config):
        # Initialize distributed training
        device_name = get_device_name()
        local_rank, rank, world_size = initialize_global_process_group()
        device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        dp_size = world_size // config.ulysses_sequence_parallel_size
        ulysses_device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))

        # Build tokenizer and datasets
        local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
        train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
        val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

        # Initialize parent class
        super().__init__(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)

    @staticmethod
    def process_trajectories(trajectories: list[Trajectory], reward_threshold: float):
        """Process trajectories into SFT format."""
        sft_data = []

        for traj in trajectories:
            if not traj:
                continue

            reward = traj.reward

            if reward < reward_threshold:
                continue

            # Get chat_completions from the last step of the trajectory
            messages = None
            if traj.steps and hasattr(traj.steps[-1], "chat_completions"):
                messages = traj.steps[-1].chat_completions

            if not messages:
                continue

            clean_messages = [{"role": msg["role"], "content": str(msg["content"]).strip()} for msg in messages if isinstance(msg, dict) and msg.get("role") and str(msg.get("content", "")).strip()]

            if len(clean_messages) >= 2:
                sft_data.append({"messages": clean_messages})

        print(f"Processed {len(trajectories)} trajectories -> {len(sft_data)} valid examples")
        return sft_data

    def train(self):
        """Start training."""
        self.fit()
