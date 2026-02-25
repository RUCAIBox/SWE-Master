from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from pathlib import Path
from datasets import load_from_disk,load_dataset
import os

# os.environ["OPENAI_API_KEY"] = "xx"
os.environ["OPENAI_API_KEY"] = "xx"

# load gym dataset [R2E-Gym/R2E-Gym-Subset, R2E-Gym/R2E-Gym-Full, R2E-Gym/SWE-Bench-Verified, R2E-Gym/SWE-Bench-Lite]
# ds = load_dataset("/media/sunshuang/SWE-Bench-Verified")
ds = load_from_disk("xx/SWE-Bench-Verified_disk")

# for i in range(len(ds)):
#     if "django__django-13821" in ds[i]["docker_image"]:
#         print(i)
#         print(ds[i]["docker_image"])
#     # print(env_args.keys)
# exit()
split = 'test' # split of the dataset [train, test]


# load gym environment
env_index = 56 # index of the environment [0, len(ds)]
env_args = EnvArgs(ds = ds[env_index])
print(env_args.ds['docker_image'])
# exit()

env = RepoEnv(env_args, ip="10.106.35.101",backend="docker")
# load agent
agent_args = AgentArgs.from_yaml(Path('./src/r2egym/agenthub/config/openhands/openhands_sp_fn_try_lsp_tools.yaml'))
# define llm: ['claude-3-5-sonnet-20241022', 'gpt-4o', 'vllm/R2E-Gym/R2EGym-32B-Agent']
# agent_args.llm_name = 'openai/kimi-k2-thinking'
agent_args.llm_name = 'hosted_vllm/qwen3-coder'
# agent_args.llm_name = 'hosted_vllm/qwen3-coder-30b-a3b-instruct'

agent = Agent(name="EditingAgent", args=agent_args)
# agent.llm_base_url = 'https://api2.aigcbest.top/v1'
# agent.llm_base_url = 'https://api.moonshot.cn/v1'
agent.llm_base_url = 'http://110.100.75.62:8000/v1'
# run the agent (note: disable fn_calling for R2E-Gym agents)
output = agent.run(env, max_steps=100, use_fn_calling=False,scaffold="openhands", use_lsp=True)

