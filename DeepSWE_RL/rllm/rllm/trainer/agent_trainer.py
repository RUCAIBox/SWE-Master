from typing import Any

import ray

from rllm.data import Dataset
from rllm.trainer.verl.train_agent_ppo import train_agent


class AgentTrainer:
    """
    A wrapper class that allows users to easily train custom agents with custom environments
    without having to directly interact with the underlying training infrastructure.
    """

    def __init__(
        self,
        agent_class: type,
        env_class: type,
        agent_args: dict[str, Any] | None = None,
        env_args: dict[str, Any] | None = None,
        config: dict[str, Any] | list[str] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
    ):
        """
        Initialize the AgentTrainer.

        Args:
            agent_class: The custom agent class to use for training
            env_class: The custom environment class to use for training
            config: Configuration overrides to apply to the default config
                   Can be a dictionary with dot notation keys (e.g., {"data.train_batch_size": 8})
                   or a list of strings in the format "key=value" (e.g., ["data.train_batch_size=8"])
            train_dataset: Optional train dataset to use
            val_dataset: Optional validation dataset to use
            agent_args: Optional arguments to pass to the agent class
            env_args: Optional arguments to pass to the environment class
        """
        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args or {}
        self.env_args = env_args or {}

        self.config = config

        if train_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.train_files = train_dataset.get_verl_data_path()
        if val_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.val_files = val_dataset.get_verl_data_path()

    def train(self):
        if not ray.is_initialized():
            ray.init(runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"},
                "pip": ["sentence_transformers"],})

        ray.get(train_agent.remote(self.config, self.agent_class, self.env_class, self.agent_args, self.env_args))
