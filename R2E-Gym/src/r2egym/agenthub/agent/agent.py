import os
import re
import copy
import yaml
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel
import requests

import litellm
from openai import OpenAI

from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.environment.env import RepoEnv
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from r2egym.agenthub.memory import Memory
from anthropic import Anthropic, AnthropicVertex  # Add Anthropic Vertex import
from r2egym.agenthub.tools import (
    r2egym_bash_execute_tool,
    search_tool,
    file_editor,
    finish_tool,
    str_replace_editor_tool,
    execute_bash_tool,
    submit_tool,
    lsp_tool
)
import traceback

logger = get_logger(__name__)  # Logger for this module
MAX_CONTEXT_TOKENS = 131072

##############################################################################
# AgentArgs Dataclass
##############################################################################
@dataclass
class AgentArgs:
    """Standard configuration for the Agent."""
    system_prompt: str
    instance_prompt: str
    command_files: List[Path]
    llm_name: str
    llm_base_url: Optional[str] = "http://localhost:8000/v1"
    demo_file: Optional[Path] = None
    use_demo: Optional[bool] = False
    other_args: Optional[Dict[str, Any]] = None  # Extra configurations

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AgentArgs":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)

@dataclass
class MemoryAgentArgs:
    """Configuration for an Agent that supports memory summarization."""
    system_prompt: str
    instance_prompt: str
    single_turn_summary_prompt: str
    multi_turns_summary_prompt: str
    command_files: List[Path]
    llm_name: str
    llm_base_url: Optional[str] = "http://localhost:8000/v1"
    demo_file: Optional[Path] = None
    use_demo: Optional[bool] = False
    other_args: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "MemoryAgentArgs":
        """Load memory-enabled configuration from a YAML file."""
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)

##############################################################################
# Agent Class
##############################################################################
class Agent:
    """Agent handles the behavior of the model and how it interacts with the environment."""

    def __init__(self, name: str, args: Union[MemoryAgentArgs, AgentArgs], logger=None):
        self.name = name
        self.args = args
        # Initialize logger: use provided logger or create a new one based on agent name
        if logger is None:
            self.logger = get_logger(name)
        else:
            self.logger = logger
        
        self.llm_name = args.llm_name

        # Determine the base URL for the LLM API
        self.llm_base_url = (
            os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
            if ("openai/" in self.llm_name) or ("hosted_vllm" in self.llm_name)
            else None
        )
        
        self.logger.info(f"llm base url:{self.llm_base_url}")
        self.logger.info(f"llm api_key:{os.environ['OPENAI_API_KEY']}")
        
        self.system_prompt_template = args.system_prompt
        self.instance_prompt_template = args.instance_prompt
        self.command_files = args.command_files
        self.other_args = args.other_args or {}
        self.logger.info(f"Initialized Agent: {name} with LLM: {args.llm_name}")
        
        self.max_retries = self.other_args.get("max_retries", 3)
        self.llm_timeout = self.other_args.get("timeout", 300)
        self.contenxt_id = ""


    def reset(self):
        """Reset the agent's trajectory and interaction history."""
        self.trajectory_steps = []
        self.history = []

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts the tokens for a list of messages using the litellm library.
        """
        token_count = litellm.token_counter(model=self.llm_name, messages=messages)
        self.logger.info(f"Total tokens in conversation: {token_count}")
        return token_count


    def model_summary(
        self, messages: List[Dict[str, str]], temperature: float = 0) -> Dict[str, Any]:
        """Query the LLM specifically for generating summaries (Memory management)."""
        response = None
        retries = 0
        tools = None

        # Determine specific LLM parameters for summarization if provided in other_args
        if "summary_llm_name" in self.other_args and self.other_args["summary_llm_name"]:
            summary_llm_name = self.other_args["summary_llm_name"]
        else:
            summary_llm_name = self.llm_name

        if "summary_api_key" in self.other_args and self.other_args["summary_api_key"]:
            summary_api_key = self.other_args["summary_api_key"]
        else:
            summary_api_key = os.environ["OPENAI_API_KEY"]

        if "summary_url_base" in self.other_args and self.other_args["summary_url_base"]:
            summary_url_base = self.other_args["summary_url_base"]
        else:
            summary_url_base = self.llm_base_url
    
        # Adjust max tokens based on the model type
        if "deepseek" in summary_llm_name:
            lite_llm_max_token = 8192
        else:
            lite_llm_max_token = 16384

        start_time = time.time()
        
        # Configure API key for local or remote models
        using_local = "hosted" in summary_llm_name
        if using_local:
            litellm.api_key = None
        else:
            litellm.api_key = summary_api_key

        messages_ = copy.deepcopy(messages)
        total_tokens = self._count_tokens(messages_)
        if total_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")
            raise ValueError(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")
        
        # Query with retry logic
        while retries < self.max_retries:
            try:
                kwargs = {
                    "tool_choice": "none",
                    "function_call": None,
                }
                if tools:
                    kwargs = {}
                if "o3" not in summary_llm_name and "o4" not in summary_llm_name:
                    kwargs["temperature"] = temperature

                response = litellm.completion(
                    model=summary_llm_name,
                    messages=messages_,
                    timeout=180,
                    api_base=summary_url_base,
                    max_tokens=lite_llm_max_token,
                    **kwargs,
                )
                self.logger.warning(f"Querying LLM complete")
                break
            except Exception as e:
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                retries += 1
                time.sleep(10)
                if "RateLimitError" in str(e):
                    time.sleep(20)
                if retries >= self.max_retries:
                    raise e

        exec_time = time.time() - start_time
        return response, exec_time

    def model_query(
        self, messages: List[Dict[str, str]], temperature: float = 0, use_lsp: bool = False) -> Dict[str, Any]:
        """Main method to query the LLM for agent actions."""
        response = None
        retries = 0
        tools = None
        
        # Setting max output tokens
        if "deepseek" in self.llm_name.lower() or "qwen25-32b" in self.llm_name.lower():
            lite_llm_max_token = 8192
        else:
            lite_llm_max_token = 16384

        # Configure tools based on the selected scaffold/framework
        if self.use_fn_calling:
            if self.scaffold == "r2egym":
                tools = [search_tool, file_editor, r2egym_bash_execute_tool, finish_tool]
            elif self.scaffold == "openhands" or self.scaffold == "sweagent":
                if use_lsp:
                    tools = [str_replace_editor_tool, execute_bash_tool, lsp_tool, submit_tool]
                else:
                    tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]
                
            elif "vertex" not in self.llm_name.lower():
                self.logger.warning(f"using prompt caching for {self.llm_name}")
                # Note: vertex is not supported yet for prompt caching in litellm
                # Add prompt caching for Anthropic models
                tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
                breakpoints_remaining = 3  # Maximum allowed cache breakpoints
                for message in reversed(messages):
                    if message["role"] in ("user", "tool"):
                        if breakpoints_remaining > 0:
                            message["cache_control"] = {"type": "ephemeral"}
                            breakpoints_remaining -= 1
                        else:
                            break

        start_time = time.time()
        using_local = "hosted" in self.llm_name
        if using_local:
            litellm.api_key = None

        messages_ = copy.deepcopy(messages)
        total_tokens = self._count_tokens(messages_)
        if total_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")
            raise ValueError(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")
        
        # LLM completion loop with retries
        while retries < self.max_retries:
            try:
                kwargs = {
                    "tool_choice": "none",
                    "function_call": None,
                }
                if tools:
                    kwargs = {"tool_choice": "auto"}
                if "o3" not in self.llm_name and "o4" not in self.llm_name:
                    kwargs["temperature"] = temperature
                    
                response = litellm.completion(
                    model=self.llm_name,
                    tools=tools,
                    messages=messages_,
                    timeout=360,
                    api_base=self.llm_base_url,
                    max_tokens=lite_llm_max_token,
                    **kwargs,
                )
                self.logger.warning(f"Querying LLM complete")
                break
            except Exception as e:
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                retries += 1
                time.sleep(10)
                if "RateLimitError" in str(e):
                    time.sleep(20)
                if retries >= self.max_retries:
                    raise e

        exec_time = time.time() - start_time
        return response, exec_time

    def parse_response(self, response: Dict[str, Any]) -> Tuple[str, Action]:
        """
        Extracts thought and action from standard XML-like formatted text.
        - thought: first <think>...</think> block
        - action: first <function=...></function> block
        """
        pattern_thought = re.compile(r"(?s)(<think>.*?</think>)")
        pattern_action = re.compile(r"(?s)(<function=.*?</function>)")
        match_thought = pattern_thought.search(response)
        match_action = pattern_action.search(response)

        if match_thought:
            thought = match_thought.group(1)
        else:
            thought = ""
        if match_action:
            action = match_action.group(1)
        else:
            action = ""
        
        thought = thought.strip()
        action = action.strip()

        # Convert the XML string to an Action object
        action = Action.from_string(action)

        return thought, action

    def parse_response_v2(self, response_text: str) -> Tuple[str, Action]:
        """
        Alternative parser:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        """
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action = match.group(1)
            thought = response_text[: match.start()]
        else:
            thought = response_text
            action = ""

        thought = thought.strip()
        action = action.strip()
        action = Action.from_string(action)

        return thought, action

    def reasoning_parser(self, response):
        """Parse reasoning content and tool calls specifically for models like Kimi."""
        think = getattr(response.choices[0].message, "reasoning_content", "")
        content = response.choices[0].message.content
        if not content:
            thought =  "<think>" + think + "</think>"
        else:
            thought =  "<think>" + think + content + "</think>"
        if not thought:
            thought = ""

        try:
            function_name = response.choices[0].message.tool_calls[0].function.name
            parameters = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            action = Action(function_name=function_name, parameters=parameters)
        except:
            action = Action(function_name="", parameters={})

        return thought, action

    def custom_parser(self, response):
        """Parse standard OpenAI-style tool calls."""
        thought = response.choices[0].message.content
        if not thought:
            thought = ""

        try:
            function_name = response.choices[0].message.tool_calls[0].function.name
            parameters = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            action = Action(function_name=function_name, parameters=parameters)
        except:
            action = Action(function_name="", parameters={})

        return thought, action

    def run(
        self,
        env: "RepoEnv",
        use_fn_calling: bool = True,
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        max_token_limit: int = 131072,
        max_exec_time: int = 90,
        max_total_time: int = 50000,
        max_llm_time: int = 7200,
        temperature=0,
        metadata: Optional[Dict[str, Any]] = {},
        scaffold: str = "r2egym",
        use_lsp: bool = False,
        use_demo: str = "",
        # Parameters for history compression and memory
        enable_compression: bool = True,     # Enable memory compression
        summary_window: int = 10,            # Window size for summarization
        keep_recent: int = 5,                # Keep the complete details of the last 5 turns
        compression_trigger_step: int = 15,  # Steps before triggering compression
        use_single_turn_summary: bool = False, 
        memory_output_path: Optional[Path] = None,
    ):
        """Main execution loop for the agent within the environment."""
        assert scaffold in ["r2egym", "openhands", "sweagent"], "Scaffold must be either r2egym, openhands or sweagent"
        self.scaffold = scaffold
        start_time = time.time()
        self.llm_timeout = max_llm_time

        # Check if the model supports native function calling
        support_fn_calling = (
            "gpt" in self.llm_name
            or "sonnet" in self.llm_name
            or "nbgexp" in self.llm_name
            or "minimax" in self.llm_name.lower()
            or "kimi" in self.llm_name.lower()
            or "deepseek" in self.llm_name.lower()
            or "devstral" in self.llm_name.lower()
            or "o3" in self.llm_name
            or "o4" in self.llm_name
            or "qwen3-max" in self.llm_name
        )
        self.use_fn_calling = use_fn_calling and support_fn_calling
        self.logger.warning(f"Using fn calling: {self.use_fn_calling}")

        self.logger.info(f"Running agent {self.name} in environment {env}.")

        # Setup environment: add predefined commands and initialize LSP if needed
        env.add_commands(self.command_files)
        if use_lsp:
            env.initialize_lsp()
        self.reset()

        # Retrieve problem statement from the environment task
        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem Statement: {problem_statement}")
        
        system_prompt = self.system_prompt_template
        self.logger.info(f"System Prompt: {system_prompt}")
        
        # Determine the base commit hash for reference
        if "base_commit" in env.runtime.ds:
            base_commit = env.runtime.ds['base_commit']
        elif "commit_hash" in env.runtime.ds:
            base_commit = env.runtime.ds['commit_hash']
        else:
            base_commit = ""

        # Construct the instance-specific prompt
        user_prompt = self.instance_prompt_template.format(
            problem_statement=problem_statement,
            working_dir='/testbed',
            base_commit=base_commit,
            test_patch_hint=metadata.get("test_patch_hint", ""),
            candidate_patch=metadata.get("candidate_patch", ""),
            candidate_patch_correctness=(
                "correct"
                if metadata.get("candidate_patch_correctness", False)
                else "incorrect"
            ),
        )

        # Prepend demo interactions if provided
        if use_demo:
            with open(use_demo, "r") as file:
                demo = file.read()
            user_prompt = f"{demo}\n\n{user_prompt}"
            self.logger.info(f"User Prompt with demo: {user_prompt}")
        else:
            self.logger.info(f"No demo, User Prompt: {user_prompt}")

        if enable_compression:
            # Initialize Memory manager for history compression
            self.memory = Memory(
                single_turn_summary_prompt=self.args.single_turn_summary_prompt,
                multi_turns_summary_prompt=self.args.multi_turns_summary_prompt,
                problem_statement=problem_statement,
                model_query_fn=self.model_summary,
                logger=self.logger,
                enable_compression=enable_compression,
                summary_window=summary_window,
                keep_recent=keep_recent,
                compression_trigger_step=compression_trigger_step,
                use_single_summary=use_single_turn_summary, 
            )

        # Initialize the conversation history
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        obs = None
        done = False
        step_count = 0
        total_time_traj = 0
        self.trajectory_steps: List[TrajectoryStep] = []


        # Main Agent Loop
        while not done:
            # Track steps remaining
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                stepcount_message = (
                            f"This is step {step_count} of a maximum of {max_steps}. "
                            f"Steps Remaining: {steps_remaining}."
                        )
            else:
                stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
            self.logger.info(stepcount_message)

            # Perform compression check after each turn to save context space
            if enable_compression and self.trajectory_steps:
                try:
                    history = self.memory.compress_history(
                        current_step=step_count,
                        trajectory_steps=self.trajectory_steps,
                        history=self.history
                    )
                    self.history = history

                    stats = self.memory.get_stats()
                    self.logger.info(
                        f"Memory stats: Compressed {stats['last_compressed_step']} steps, "
                        f"{stats['num_single_summaries']} single summaries, "
                        f"{stats['num_aggregated_summaries']} agg summaries"
                    )
                except Exception as e:
                    self.logger.error(f"Memory compression failed: {e}")


            # Query the LLM for the next action
            messages = copy.deepcopy(self.history)
            model_gen_max_num= 3
            model_gen_finished_flag = False
            for model_gen_try in range(model_gen_max_num):
                try:
                    response, llm_exec_time = self.model_query(messages, temperature, use_lsp)
                    model_gen_finished_flag = True
                    break
                except Exception as e:
                    self.logger.error(f"Error querying LLM: {e}. Attempt {model_gen_try+1} failed.")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")

            if not model_gen_finished_flag:
                done = True
                exit_reason = "llm_query_error"
                break

            # Process token usage stats
            if hasattr(response, "usage"):
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)
                self.logger.info(
                    f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                )
            else:
                completion_tokens = -1
                prompt_tokens = -1
                total_tokens = self._count_tokens(messages)
                self.logger.warning("No token usage information available.")

            # Fix specific formatting issues for Qwen model output here
            response_content = response.choices[0].message.content
            if response_content and "<parameter=command=" in response_content:
                # Correction for Qwen improperly replacing '=' with '>' at the end of a parameter
                self.logger.warning(f"Fixing Qwen parameter formatting error.")
                response_content = response_content.replace("<parameter=command=","<parameter=command>")
                response.choices[0].message.content = response_content

            self.response = response 
            assistant_message = response.choices[0].message.content
            self.logger.info(f"Assistant's message:\n{assistant_message}\n")
            
            # Select appropriate parser for the response
            if self.use_fn_calling:
                if "kimi" in self.llm_name:
                    thought, action = self.reasoning_parser(response)
                else:
                    thought, action = self.custom_parser(response)
            elif "qwen" in self.llm_name or "kat" in self.llm_name:
                thought, action = self.parse_response_v2(assistant_message)
            else:
                thought, action = self.parse_response(assistant_message)

            self.logger.info(f"THOUGHT:\n{thought}\n")
            self.logger.info(f"ACTION:\n{action}\n")

            # Execute the action in the environment
            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
            except Exception as e:
                obs = str(e)
                self.logger.error(f"Error during environment step: {obs}")

            env_exec_time = info["total_time"]
            total_step_time = llm_exec_time + env_exec_time
            total_time_traj += total_step_time
            step_count += 1 

            # Update history with assistant message and environment observation
            if self.use_fn_calling:
                assistant_response = response.choices[0].message.dict()
                if assistant_response.get("tool_calls", None):
                    assistant_response["tool_calls"] = assistant_response["tool_calls"][:1]
                self.history.append(assistant_response)

                obs = str(obs)
                obs_with_budget_aware = f"{obs}\n{stepcount_message}"
                
                try:
                    function_name = response.choices[0].message.tool_calls[0].function.name
                    function_id = response.choices[0].message.tool_calls[0].id
                    self.history.append({
                        "role": "tool",
                        "content": str(obs_with_budget_aware),
                        "name": function_name,
                        "tool_call_id": function_id,
                    })
                except Exception as e:
                    self.logger.error(f"Error logging tool response: {e}")
                    self.history.append({"role": "user", "content": str(obs)})
            else:
                assistant_message = f"{thought}\n\n{action.to_xml_string()}"
                self.history.append({"role": "assistant", "content": assistant_message})
                obs = str(obs)
                obs_with_budget_aware = f"{obs}\n{stepcount_message}"
                self.history.append({"role": "user", "content": str(obs_with_budget_aware)})


            self.logger.info(f"OBSERVATION:\n{obs_with_budget_aware}\n")
            self.logger.info("-" * 50)

            # Determine exit reasons
            if done:
                if steps_remaining > 0:
                    self.logger.info("Agent has finished naturally.")
                    exit_reason = "agent"
                elif steps_remaining == 0:
                    self.logger.info("Agent reached max steps.")
                    exit_reason = "max_step_limit"
                else:
                    exit_reason = "agent_max_step_limit"
            elif total_tokens >= max_token_limit:
                self.logger.info("Agent reached token limit.")
                exit_reason = "token_limit"
                done = True
            elif step_count >= max_steps_absolute:
                self.logger.info("Agent reached absolute step limit.")
                exit_reason = "abs_step_limit"
                done = True
            elif total_time_traj >= max_total_time:
                self.logger.info("Agent reached total time limit.")
                exit_reason = "traj_time_limit"
                done = True

            # Record trajectory step
            trajectory_step = TrajectoryStep(
                step_idx=step_count - 1,
                thought=thought,
                action=action.to_xml_string(),
                observation=str(obs_with_budget_aware),
                done=done,
                info=info,
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_total=total_tokens,
                llm_exec_time=llm_exec_time,
                env_exec_time=env_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
            )
            self.trajectory_steps.append(trajectory_step)

        # Retrieve the final patch (diff) after the agent finishes
        output_patch = env.runtime.get_patch()
        instance_id =  env.runtime.ds['instance_id']

        # Export memory logs if requested
        if memory_output_path and self.memory:
            try:
                memory_output_path_obj = Path(memory_output_path)
                self.memory.export_memory(history=self.history, instance_id=instance_id, output_path=memory_output_path_obj)
                self.logger.info(f"Memory exported to: {memory_output_path}")

                # Ensure parameters saved in the trajectory match the execution parameters
                if "default_summary_window" in self.args.other_args:
                    self.args.other_args['default_summary_window'] = summary_window
                
                if "default_keep_recent" in self.args.other_args:
                    self.args.other_args['default_keep_recent'] = keep_recent
                
                if "default_compression_trigger" in self.args.other_args:
                    self.args.other_args['default_compression_trigger'] = compression_trigger_step

            except Exception as e:
                self.logger.error(f"Failed to export memory: {e}")

        # Construct final Trajectory object
        self.trajectory = Trajectory(
            trajectory_steps=[traj_step.model_dump() for traj_step in self.trajectory_steps],
            problem_statement=problem_statement,
            docker_image=env.runtime.docker_image,
            agent_args=asdict(self.args),
            env_args=asdict(env.args),
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=max_llm_time,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            exit_reason=exit_reason,
            output_patch=output_patch,
        )

        self.logger.info(f"Agent completed in {time.time() - start_time} seconds.")
        return self.trajectory