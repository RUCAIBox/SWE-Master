<h1 align="center"> DeepSWE-Preview - Training a State-of-the-Art Coding Agent by Scaling RL </h1>

<!-- paper . data and models . project page -->
<p align="center">
<a href="#">📃 Blog Post</a>
•
<a href="https://huggingface.co/agentica-org/DeepSWE-Preview" > 🤗 Data & Models</a>
•
<!-- project page -->
<a href="https://wandb.ai/mluo/deepswe" >🔥 WandB Logs</a>
•
<a href="https://drive.google.com/file/d/10LIwpJeaFuiX6Y-qEG2a4a335PEuQJeS/view?usp=sharing" > 📈 Evaluation Logs</a>
•
<a href="https://agentica-project.com/" > 🌐 Project Page</a>
•
<a href="https://github.com/agentica-project/rllm" > 🧑‍💻 Code</a>
</p>

<div align="center">

[![Github](https://img.shields.io/badge/RLLM-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/agentica-project/rllm)
[![Website](https://img.shields.io/badge/Site-%23000000.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://www.agentica-project.com) 
[![Twitter](https://img.shields.io/badge/Agentica-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/Agentica_)
[![Hugging Face Collection](https://img.shields.io/badge/Agentica-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/agentica-org)

</div>

We introduce **`DeepSWE-Preview`**, a reasoning-enabled coding agent trained from scratch from `Qwen3-32B` with only reinforcement learning (RL). It achieves 59.2**%** on SWE-Bench-Verified with test-time scaling, reaching SOTA for open-weight coding agents  (**42.2%** Pass@1, **71.0%** Pass@16).

DeepSWE is trained using [**rLLM**](https://www.notion.so/21b81902c146819db63cd98a54ba5f31?pvs=21), our framework for post-training language agents. We’ve **open sourced** everything—our dataset, code, training, and eval logs, for everyone to progress on scaling and improving agents with RL.


## Getting Started 🎯

### Installation
```bash
# Installing Python 3.10 Environment.
conda create -n rllm python=3.10 -y
conda activate rllm

# Installing RLLM dependencies.
cd rllm
pip install -e ./verl
pip install -e .
```

Also, install R2E-Gym for high-quality SWE-Bench environments used for RL training.
```bash
git clone https://github.com/agentica-project/R2E-Gym.git
cd R2E-Gym
pip install -e .
```


### Data and Agent Scaffold

We use the R2E-Gym environments for RL training. R2E-Gym environment can be simply used as:
```python
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from pathlib import Path
from datasets import load_dataset

# load gym dataset [R2E-Gym/R2E-Gym-Subset, R2E-Gym/R2E-Gym-Full, R2E-Gym/SWE-Bench-Verified, R2E-Gym/SWE-Bench-Lite]
ds = load_dataset("R2E-Gym/R2E-Gym-Lite")
split = 'train' # split of the dataset [train, test]

# load gym environment
env_index = 100 # index of the environment [0, len(ds)]
env_args = EnvArgs(ds = ds[split][env_index])
env = RepoEnv(env_args)

# load agent
agent_args = AgentArgs.from_yaml(Path('./src/r2egym/agenthub/config/edit_fn_calling.yaml'))
# define llm: ['claude-3-5-sonnet-20241022', 'gpt-4o', 'vllm/R2E-Gym/R2EGym-32B-Agent']
agent_args.llm_name = 'claude-3-5-sonnet-20241022'
agent = Agent(name="EditingAgent", args=agent_args)

# run the agent (note: disable fn_calling for R2E-Gym agents)
output = agent.run(env, max_steps=40, use_fn_calling=True)
```

### Inference using DeepSWE-Preview

First, launch a [DeepSWE-Preview](https://huggingface.co/agentica-org/DeepSWE-Preview) using vllm or sglang.
```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve agentica-org/DeepSWE-Preview   --tensor-parallel-size 8   --max-model-len 70000   --hf-overrides '{"max_position_embeddings": 70000}'
```

Run inference using DeepSWE-Preview and [R2E-Agent](https://github.com/R2E-Gym/R2E-Gym)
```
python -m r2egym.agenthub.run.edit runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/agentica-org/DeepSWE-Preview"   --use_fn_calling False   --exp_name deepswe-preview-eval-v1 --temperature 1   --max_steps_absolute 100   --backend "docker"
```


**Trajectory Visualization:** The generated trajectories are saved in `./traj` directory. You can visualize the trajectories using,
```bash
python app/app.py --traj_dir "./traj"
```


### Hybrid Test-time Scaling

We adopt the hybrid test-time scaling approach from [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) for scaling test-time compute.

First collect the desired number of rollouts for scaling test-time compute.
```bash
N=16 # number of rollouts to collect

for i in {1..N}; do
    python -m r2egym.agenthub.run.edit runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/agentica-org/DeepSWE-Preview"   --use_fn_calling False   --exp_name deepswe-preview-eval-v1-rollout$i --temperature 1   --max_steps_absolute 100   --backend "docker"
done
```

For ease of use, we already provide 16 rollouts here: [Eval Logs](https://drive.google.com/file/d/10LIwpJeaFuiX6Y-qEG2a4a335PEuQJeS/view?usp=sharing)


The hybrid test-time scaling consists of three steps:

#### Execution-free Verifier

Prepare the data in execution-free verifier format.
```bash
python verifier_data_prep.py create --push_to_hub=True --verifier_traj_dir="./traj" --hub_repo_name="deepswe-verifier-debug-condense-v1" --max_workers=54 --filter_method="agent_priority"
```

For ease of use, we already provide the verifier dataset here: [Verifier Dataset](https://huggingface.co/datasets/r2e-edits/deepswe-verifier-debug-condense-v1)


Run the execution-free verifier on the collected rollouts.
```bash
# launch a vllm server for the verifier
vllm serve Qwen/Qwen3-14B --max-model-len 76800 --hf-overrides '{"max_position_embeddings": 76800}' --enable-lora --lora-modules a=/home/ubuntu/360-LLaMA-Factory/output/verifier --port 8000 --dtype bfloat16 --max-lora-rank 64 --tensor-parallel-size 8  

# run the verifier
python verifier_eval.py --llm_name "hosted_vllm/a" --eval_dataset "r2e-edits/deepswe-swebv-eval-n16-verifier-v1" --out_file 'deepswe-verifier.csv'
```

For ease of use, we already provide the verifier output here: [Verifier Output](https://drive.google.com/file/d/10LIwpJeaFuiX6Y-qEG2a4a335PEuQJeS/view?usp=sharing)

Compute the accuracy with EF verifier.
```bash
python verifier_analyze.py --dir "./verifier_traj"
```

#### Execution-based Verifier


#### Hybrid Test-time Scaling




## Citation

```
@misc{deepswe2025,
  title={DeepSWE: Training a State-of-the-Art Coding Agent from Scratch by Scaling RL},
  author={Michael Luo, Naman Jain, Jaskirat Singh, Sijun Tan, Ameen Patel, Qingyang Wu, Alpay Ariyak, Colin Cai, Tarun Venkat, Shang Zhu, Ben Athiwaratkun, Manan Roongta, Ce Zhang, Li Erran Li, Raluca Ada Popa, Koushik Sen, Ion Stoica},
  howpublished={\url{N/A}},
  note={Notion Blog},
  year={2025}
}
```