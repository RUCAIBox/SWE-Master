"""
EditAgent script for running agents on Docker images.

This module provides functions to run EditAgent on single or multiple Docker images,
managing Docker image pulling, environment setup, agent execution, and result collection.
"""

import concurrent.futures
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import docker
from datasets import load_dataset, load_from_disk
from fire import Fire

from r2egym.agenthub.agent.agent import Agent, AgentArgs, MemoryAgentArgs
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.trajectory import Trajectory
from r2egym.agenthub.utils.log import get_logger
from r2egym.docker_bash_utils.docker_list_tags import fetch_docker_tags
from r2egym.logging import INFO, setup_logging

# Initialize logger for this module
logger = get_logger(__name__)

# File lock for thread-safe writing to output files
file_lock = threading.Lock()

# Timeout for agent execution in seconds
TIMEOUT_SECONDS = 1200

##############################################################################
def get_docker_images(repo_name: str) -> List[str]:
    """
    Fetches the list of Docker images available for the base image.

    Args:
        repo_name: Name of the repository to fetch Docker images for.

    Returns:
        A list of Docker image tags in the format 'namanjain12/{repo_name}new:tag'.
    """
    base_image = f"namanjain12/{repo_name}new"
    tags = fetch_docker_tags(base_image)
    docker_image_list = [f"{base_image}:{x['name']}" for x in tags]
    return docker_image_list


def prepull_docker_image(docker_image: str) -> bool:
    """
    Prepulls a single Docker image.
    
    Args:
        docker_image: The Docker image name to pull
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = docker.from_env()

        # existing_images = client.images.list(name=docker_image.split(":")[0])
        
        
        # Check if the image already exists locally
        try:
            client.images.get(docker_image)
            logger.info(f"Docker image already exists locally: {docker_image}")
            return True
        except docker.errors.ImageNotFound:
            # Image doesn't exist locally, proceed to pull
            logger.info(f"Docker image not found locally, pulling: {docker_image}")
            client.images.pull(docker_image)
            logger.info(f"Successfully pulled Docker image: {docker_image}")
            return True
        except Exception as e:
            logger.error(f"Error checking for existing Docker image {docker_image}: {e}")
            return False
    except Exception as e:
        logger.error(f"Failed to pull Docker image {docker_image}: {e}")
        return False


def prepull_docker_images(ds_selected: List[Dict], max_workers: Optional[int] = None, ip: str = "") -> None:
    """
    Prepulls all Docker images in parallel before starting the main execution.

    Args:
        ds_selected: List of dataset entries containing docker_image keys.
        max_workers: Maximum number of threads for parallel pulling.
        ip: IP address of the Docker daemon.
    """
    # Extract unique Docker images from the dataset entries
    docker_images = list(set([ds_entry["docker_image"] for ds_entry in ds_selected]))
    logger.info(f"Starting parallel prepull of {len(docker_images)} unique Docker images...")

    # Use ThreadPoolExecutor for I/O bound operations like Docker pulls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pull tasks
        future_to_image = {
            executor.submit(prepull_docker_image, docker_image, ip): docker_image
            for docker_image in docker_images
        }

        # Track results
        successful_pulls = []
        failed_pulls = []

        for future in concurrent.futures.as_completed(future_to_image):
            docker_image = future_to_image[future]
            try:
                success = future.result()
                if success:
                    successful_pulls.append(docker_image)
                else:
                    failed_pulls.append(docker_image)
            except Exception as e:
                logger.error(f"Exception during prepull of {docker_image}: {e}")
                failed_pulls.append(docker_image)

    logger.info(f"Prepull completed. Success: {len(successful_pulls)}, Failed: {len(failed_pulls)}")
    if failed_pulls:
        logger.warning(f"Failed to pull images: {failed_pulls}")


##############################################################################
# editagent Functions
##############################################################################
def run_agent_with_restarts(
    agent: Agent,
    env: RepoEnv,
    max_steps=40,
    num_restarts=1,
    temperature=0.0,
    max_steps_absolute=50,
    use_fn_calling: bool = True,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 131072,
    use_lsp: bool = False,
    use_demo: str = "",
    enable_compression: bool = False,     # Whether to enable memory compression
    summary_window: int = 10,
    keep_recent: int = 5,                # Keep full details for the most recent 5 turns
    compression_trigger_step: int = 20,  # Step at which compression is triggered
    use_single_turn_summary: bool = False,
    # Memory output path
    memory_output_path: Optional[Path] = None,
) -> Trajectory:
    """
    Iterative evaluation protocol for agent execution with restarts.

    The protocol runs the agent multiple times with different configurations:
    - Normally run the agent
    - Run for maximum num_iterations times
    - Stop if trajectory.exit_reason == "agent"
    - Otherwise continue iteratively until maximum iterations
    - Finally choose the trajectory with the lowest number of steps
    - Note: restarts and iterative_evals are different (only one can be used)
    - If original temperature is 0, use increasing temperatures up to 0.2

    Args:
        agent: The agent instance to run.
        env: The environment instance.
        max_steps: Maximum total steps across all restarts/iterations.
        num_restarts: Number of restarts per iteration.
        temperature: Initial temperature for LLM sampling.
        max_steps_absolute: Absolute maximum steps per agent run.
        use_fn_calling: Whether to use function calling.
        max_iterations: Maximum number of iterations.
        scaffold: Scaffold type ("r2egym", "sweagent", "openhands").
        max_tokens: Maximum token limit for the agent.
        use_lsp: Whether to use Language Server Protocol.
        use_demo: Demo string for few-shot learning.
        enable_compression: Whether to enable memory compression.
        summary_window: Window size for memory summarization.
        keep_recent: Number of recent turns to keep in full detail.
        compression_trigger_step: Step count at which compression triggers.
        use_single_turn_summary: Whether to use single-turn summarization.
        memory_output_path: Optional path to save memory output.

    Returns:
        The selected trajectory with the lowest number of steps.
    """
    steps_per_agent = max_steps // num_restarts
    logger.warning(f"running {steps_per_agent} steps per agent")

    # only one of restarts > 1 and iterative_eval can be True
    iterative_eval = max_iterations > 1
    assert not (num_restarts > 1 and iterative_eval), "only one of restarts > 1 and iterative_eval can be True"
    logger.warning(f"Using iterations: {max_iterations}, using iterative protocol: {iterative_eval}")

    # if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    # if temperature is 0, create list of increasing temperatures up to 0.2
    if temperature == 0:
        temperatures = [0.0 + 0.1 * i for i in range(max_iterations)]
        temperatures = [min(t, 0.2) for t in temperatures]  # cap at 0.2
    else:
        temperatures = [temperature] * max_iterations
    logger.warning(f"Using temperatures: {temperatures}")

    # run the agent in iterative protocol
    trajectories = []
    for iteration in range(max_iterations):
        for idx in range(num_restarts):
            logger.warning(f"running agent at idx: {idx+1}")
            trajectory = agent.run(
                env,
                max_steps=steps_per_agent,
                temperature=temperatures[iteration],
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                scaffold=scaffold,
                max_token_limit=max_tokens,
                use_lsp=use_lsp,
                use_demo=use_demo,
                enable_compression=enable_compression,     # Whether to enable memory compression
                summary_window=summary_window,
                keep_recent=keep_recent,                # Keep full details for the most recent 5 turns
                compression_trigger_step=compression_trigger_step,  # Step at which compression triggers
                use_single_turn_summary=use_single_turn_summary,
                # Memory output path
                memory_output_path=memory_output_path,
            )
            # remove reproduce.py
            # env.runtime.run('rm reproduce_issue.py')
        if trajectory.exit_reason == "agent":
            logger.warning(f"agent self-finished at iteration: {iteration}")
            return trajectory
        # otherwise continue iteratively
        trajectories.append(trajectory)
        # reset the env
        # env.reset()

    # shutdown lsp daemon
    if use_lsp:
        bash_output, error_code = env.runtime.run("ls /var/tmp")
        logger.info(f"After run for ls /var/tmp output: \n {bash_output}, \n error_code: \n {error_code}")
        bash_output, error_code = env.runtime.run("cat /var/tmp/lsp_port_session_abc.pid")
        logger.info(f"After run for cat /var/tmp/lsp_port_session_abc.pid output: \n {bash_output}, \n error_code: \n {error_code}")

        bash_output, error_code = env.runtime.run("cat /usr/local/bin/daemon.log")
        logger.info(f"After run for cat /usr/local/bin/daemon.log output: \n {bash_output}, \n error_code: \n {error_code}")
        bash_output, error_code = env.runtime.run("cat /usr/local/bin/daemon_stdout.log")
        logger.info(f"After run for cat /usr/local/bin/daemon_stdout.log output: \n {bash_output},\n error_code: \n {error_code}")
        bash_output, error_code = env.runtime.run("ps aux | grep lsp_daemon")
        logger.info(f"After run for ps aux | grep lsp_daemon output: \n {bash_output},\n error_code: \n {error_code}")
        bash_output, error_code = env.runtime.run("ps -ef | grep lsp_daemon")
        logger.info(f"After run for ps -ef | grep lsp_daemon output: \n {bash_output},\n error_code: \n {error_code}")
        env.runtime.run("lsp_tool daemon_shutdown")

    # choose the trajectory with the lowest number of steps
    trajectory = min(trajectories, key=lambda x: x.num_steps)
    return trajectory

def runagent(
    ds: Dict,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    max_steps: int = 40,
    num_restarts: int = 1,
    max_steps_absolute: int = 50,
    llm_name: str = "gpt-4o",
    temperature: float = 0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes",  # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 131072,
    ip: str = "",
    use_lsp: bool = False,
    used_yaml: str = "",
    use_demo: str = "",
    enable_compression: bool = False,     # Whether to enable memory compression
    summary_window: int = 10,
    keep_recent: int = 5,                # Keep full details for the most recent 5 turns
    compression_trigger_step: int = 20,  # Step at which compression triggers
    use_single_turn_summary: bool = False,
    # Memory output path
    memory_output_path: Optional[Path] = None,
) -> Optional[str]:
    """
    Runs the editagent agent on a specified dataset entry.

    Args:
        ds: Dataset entry containing docker_image, instance_id, and problem_statement.
        traj_dir: Directory to save trajectories.
        exp_name: Experiment name used for logging and file naming.
        max_steps: Maximum total steps across all restarts/iterations.
        num_restarts: Number of restarts per iteration.
        max_steps_absolute: Absolute maximum steps per agent run.
        llm_name: Name of the LLM to use.
        temperature: Temperature for LLM sampling.
        use_fn_calling: Whether to use function calling.
        backend: Backend type ("kubernetes" or "docker").
        max_reward_calc_time: Maximum time to calculate reward.
        max_iterations: Maximum number of iterations.
        scaffold: Scaffold type ("r2egym", "sweagent", "openhands").
        max_tokens: Maximum token limit for the agent.
        ip: IP address of the Docker daemon.
        use_lsp: Whether to use Language Server Protocol.
        used_yaml: Path to custom YAML configuration file.
        use_demo: Demo string for few-shot learning.
        enable_compression: Whether to enable memory compression.
        summary_window: Window size for memory summarization.
        keep_recent: Number of recent turns to keep in full detail.
        compression_trigger_step: Step count at which compression triggers.
        use_single_turn_summary: Whether to use single-turn summarization.
        memory_output_path: Optional path to save memory output.

    Returns:
        JSON string of the trajectory if successful, None otherwise.
    """
    logger = setup_logging(
        name=ds["docker_image"].replace("/", "_"),
        log_file=f"run_logs/{exp_name}/{ds['docker_image'].replace('/', '_')}_{ds['instance_id']}.log",
        console=True,
        level=INFO,
    )
    logger.info(f"Starting editagent on Docker image: {ds['docker_image']}")
    logger.info(f"Using LLM: {llm_name}")
    logger.info(f"Max Steps: {max_steps}")

    assert scaffold in ["r2egym", "sweagent", "openhands"], f"Scaffold is {scaffold}, must be one of [r2egym, sweagent, openhands]"
    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize environment arguments
    env_args = EnvArgs(ds=ds)

    # Initialize the RepoEnv
    env = RepoEnv(env_args, ip=ip, logger=logger, backend=backend, use_lsp=use_lsp)
    logger.info("has set up env")
    # Set agent arguments
    # Enable memory compression
    if enable_compression:
        if used_yaml:
            agent_args = MemoryAgentArgs.from_yaml(
                Path(used_yaml)
            )
        else:
            if use_fn_calling:
                assert scaffold != "sweagent", "SWEagent scaffold does not support fn calling"
                agent_args = MemoryAgentArgs.from_yaml(
                    Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml")
                )
            else:
                if "openhands" in scaffold:
                    agent_args = MemoryAgentArgs.from_yaml(
                        Path(f"./src/r2egym/agenthub/config/{scaffold}/openhands_sp_non_fn_calling.yaml")
                    )
                else:
                    agent_args = MemoryAgentArgs.from_yaml(
                        Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
                    )
    # Normal rollout (without memory compression)
    else:
        if used_yaml:
            agent_args = AgentArgs.from_yaml(
                Path(used_yaml)
            )
        else:
            if use_fn_calling:
                assert scaffold != "sweagent", "SWEagent scaffold does not support fn calling"
                agent_args = AgentArgs.from_yaml(
                    Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml")
                )
                
            else:
                if "openhands" in scaffold:
                    agent_args = AgentArgs.from_yaml(
                        Path(f"./src/r2egym/agenthub/config/{scaffold}/openhands_sp_non_fn_calling.yaml")
                    )
                else:
                    agent_args = AgentArgs.from_yaml(
                        Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
                    )

    logger.info("before agent")  
    agent_args.llm_name = llm_name

    # Initialize the agent
    agent = Agent(name="EditAgent", args=agent_args, logger=logger)
    logger.info("after agent")
    # run agent editagent
    try:
        trajectory = run_agent_with_restarts(
            agent,
            env,
            max_steps=max_steps,
            num_restarts=num_restarts,
            temperature=temperature,
            max_steps_absolute=max_steps_absolute,
            use_fn_calling=use_fn_calling,
            max_iterations=max_iterations,
            scaffold=scaffold,
            max_tokens=max_tokens,
            use_lsp=use_lsp,
            use_demo=use_demo,
            enable_compression=enable_compression,     # Whether to enable memory compression
            summary_window=summary_window,
            keep_recent=keep_recent,                # Keep full details for the most recent 5 turns
            compression_trigger_step=compression_trigger_step,  # Step at which compression triggers
            use_single_turn_summary=use_single_turn_summary,
            # Memory output path
            memory_output_path=memory_output_path,
        )
    except Exception as e:
        logger.error(
            f"Error during agent run for Docker image {ds['docker_image']}: {e}"
        )
        return None

    # also get the gt outputs
    reward_calc_time = time.time()
    reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
    reward_calc_time = time.time() - reward_calc_time
    # Close the environment and runtime
    env.close()

    # update the trajectory object
    trajectory.reward = reward
    trajectory.test_output = test_output
    trajectory.ds = ds
    trajectory.exp_name = exp_name
    trajectory.reward_calc_time = reward_calc_time # time taken to calculate reward
    logger.warning(f"time taken to calculate reward in seconds: {reward_calc_time:.2f}")

    logger.info(f"editagent completed for Docker image: {ds['docker_image']}")
    # close env and docker runtime
    logger.info(f"Closing environment for Docker image: {ds['docker_image']}")

    json_traj = trajectory.model_dump_json()
    logger.info(f"has processed json_traj for docker image {ds['docker_image']}")
    return json_traj


def runagent_multiple(
    dataset: str,
    split: str,
    k: int = 1,
    force_k: bool = False,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx: int = 0,
    max_steps: int = 40,
    num_restarts: int = 1,
    max_steps_absolute: int = 50,
    max_workers: Optional[int] = None,
    llm_name: str = "gpt-4o",
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 0,
    use_fn_calling: bool = True,
    backend: str = "docker",  # "kubernetes" or "docker"
    max_reward_calc_time: int = 1800,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    prepull_images: bool = False,
    max_tokens: int = 131072,
    ip: str = "",
    used_yaml: str = "",
    # ----- LSP parameters -----
    use_lsp: bool = False,  # Whether to enable Language Server Protocol
    use_demo: str = "",  # Whether to provide few-shot examples
    # ----- Summary parameters -----
    enable_compression: bool = False,     # Whether to enable memory compression
    summary_window: int = 10,
    keep_recent: int = 5,                # Keep full details for the most recent 5 turns
    compression_trigger_step: int = 20,  # Step at which compression triggers
    use_single_turn_summary: bool = False,
    memory_output_path: Optional[Path] = None,    # Memory output path
):
    """
    Runs the editagent agent on multiple dataset entries.

    Args:
        dataset: Dataset name or path to load.
        split: Dataset split to use.
        k: Number of dataset entries to process.
        force_k: If True, force processing k entries even if some exist.
        traj_dir: Directory to save trajectories.
        exp_name: Experiment name for the JSONL file.
        start_idx: Starting index in the dataset.
        max_steps: Maximum total steps across all restarts/iterations.
        num_restarts: Number of restarts per iteration.
        max_steps_absolute: Absolute maximum steps per agent run.
        max_workers: Maximum number of parallel workers.
        llm_name: Name of the LLM to use.
        use_existing: Whether to use existing results in JSONL file.
        skip_existing: Whether to skip entries already in exclude.jsonl.
        temperature: Temperature for LLM sampling.
        use_fn_calling: Whether to use function calling.
        backend: Backend type ("kubernetes" or "docker").
        max_reward_calc_time: Maximum time to calculate reward.
        max_iterations: Maximum number of iterations.
        scaffold: Scaffold type ("r2egym", "sweagent", "openhands").
        prepull_images: Whether to prepull Docker images in parallel.
        max_tokens: Maximum token limit for the agent.
        ip: IP address of the Docker daemon.
        used_yaml: Path to custom YAML configuration file.
        use_lsp: Whether to use Language Server Protocol.
        use_demo: Demo string for few-shot learning.
        enable_compression: Whether to enable memory compression.
        summary_window: Window size for memory summarization.
        keep_recent: Number of recent turns to keep in full detail.
        compression_trigger_step: Step count at which compression triggers.
        use_single_turn_summary: Whether to use single-turn summarization.
        memory_output_path: Optional path to save memory output.
    """
    # Load the dataset
    if dataset.endswith(".json"):
        with open(dataset, "r") as f:
            ds = json.load(f)
            print(f"loaded dataset from {dataset}, length: {len(ds)}")
    else:
        logger.info(f"use load_dataset")
        # ds = load_dataset(dataset, split=split)
        try:
            ds = load_dataset(dataset, split=split)
        except:
            ds = load_from_disk(dataset)
        print(ds["instance_id"][:5])
        logger.info(f"{len(ds)}, {k}, {start_idx}")
        # shuffle the dataset

    # print(ds["instance_id"][:5])
    logger.info(f"{len(ds)}, {k}, {start_idx}")
    selected_idx = range(start_idx, start_idx + k)
    ds_selected = [ds[i] for i in selected_idx]

    # shuffle the dataset
    # ds = ds.shuffle(seed=42)

    # print ds_selected stats
    logger.info(
        # f"Dataset: {dataset}, Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
        f"Dataset: , Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
    )
    logger.info(f"Starting editagent on {len(ds_selected)} Docker images.")

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure traj_dir exists
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename for the JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"

    if use_existing:
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                existing_dockers = []
                for line in f.readlines():
                    try:
                        existing_dockers.append(
                            Trajectory.load_from_model_dump_json(line).ds[
                                "problem_statement"
                            ] 
                        )
                    except:
                        print("error in jsonl file")
                
                if force_k:
                    ds_selected = [
                        ds_entry
                        for ds_entry in ds
                        if ds_entry["docker_image"] not in existing_dockers
                    ][:k]
                else:
                    ds_selected = [
                        ds_entry
                        for ds_entry in ds_selected
                        if ds_entry["problem_statement"] not in existing_dockers
                    ]

    if skip_existing:
        old_jsonl_file = traj_dir_path / "exclude.jsonl"
        with open(old_jsonl_file) as f:
            existing_dockers = [
                loadline["ds"]["docker_image"]
                for line in f
                for loadline in [json.loads(line)]
            ]

        logger.info(
                f"skip_existing {len(existing_dockers)} Docker images for path: {old_jsonl_file} after filtering."
        )

        ds_selected = [
            ds_entry
            for ds_entry in ds_selected
            if ds_entry["docker_image"] not in existing_dockers
        ]

    logger.info(
        f"Starting editagent on {len(ds_selected)} Docker images after filtering."
    )

    # Prepull all Docker images in parallel before starting main execution
    if ds_selected and prepull_images:
        logger.info("Prepulling Docker images before starting main execution...")
        prepull_docker_images(ds_selected, max_workers=max_workers, ip=ip)
        logger.info("Docker image prepull completed.")

    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
       
        future_to_image = {}
        for idx, ds_entry in enumerate(ds_selected):
            docker_image = ds_entry["docker_image"]
            docker_ip = ds_entry.get("ip","")
            if docker_ip:
                ip_used = docker_ip
                logger.info(f"{docker_image} provides its own IP, using corresponding IP: {docker_ip}")
                logger.info(f"[{idx}/{len(ds_selected)}] Submitting task: {docker_image}")
            else:
                ip_used = ip
                logger.info(f"{docker_image} entry does not contain an IP, using pre-set IP: {ip}")
                logger.info(f"[{idx}/{len(ds_selected)}] Submitting task: {docker_image}")
            fut = executor.submit(
                runagent,
                ds=ds_entry,
                traj_dir=traj_dir,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=num_restarts,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                temperature=temperature,
                use_fn_calling=use_fn_calling,
                backend=backend,
                max_reward_calc_time=max_reward_calc_time,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
                ip=ip_used,
                use_lsp=use_lsp,
                used_yaml=used_yaml,
                use_demo=use_demo,
                enable_compression=enable_compression,     # Whether to enable memory compression
                summary_window=summary_window,
                keep_recent=keep_recent,                # Keep full details for the most recent 5 turns
                compression_trigger_step=compression_trigger_step,  # Step at which compression triggers
                use_single_turn_summary=use_single_turn_summary,
                # Memory output path
                memory_output_path=memory_output_path,
            )

            future_to_image[fut] = docker_image


        with open(jsonl_file, "a") as f:
            for future in concurrent.futures.as_completed(future_to_image):
                docker_image = future_to_image[
                    future
                ]  # <-- retrieve that stored docker_image
                try:
                    result = future.result(timeout=1200)
                    if result is not None:
                        with file_lock:
                            f.write(result + "\n")
                # Catch specific TimeoutError
                except TimeoutError:
                    logger.error(f"Timeout occurred for Docker image {docker_image} after {TIMEOUT_SECONDS} seconds.")

                except Exception as e:
                    # Use docker_image from above when logging
                    logger.error(f"Can not write for Docker image {docker_image}: {e}")

    logger.info(f"editagent completed on {len(ds_selected)} Docker images.")


if __name__ == "__main__":
    # Expose functions via Fire
    Fire(
        {
            "runagent": runagent,
            "runagent_multiple": runagent_multiple,
        }
    )
