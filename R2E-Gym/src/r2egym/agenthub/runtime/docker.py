import os, sys
import json
from time import sleep
import time
from typing import Any, Optional, List, Tuple, Union
import uuid
import tempfile
import docker
from pathlib import Path
from docker.models.containers import Container

from r2egym.repo_analysis.execution_log_parser import parse_log_fn, decolor_dict_keys
from r2egym.agenthub.runtime.base import (
    ExecutionEnvironment,
)
import base64
import subprocess
import datetime
import hashlib
import shutil
import uuid

import docker
import kubernetes
import tarfile
import io
import os
from r2egym.agenthub.utils.log import get_logger
import re
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS, SKIP_FILES, SKIP_FILES_NEW, CMD_TIMEOUT
import concurrent.futures

from r2egym.agenthub.trajectory.swebench_utils import (
    make_test_spec,
    swebench_parse,
    # TestSpec,
)

from r2egym.agenthub.utils.utils import get_logger
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.swesmith.utils import get_test_command

from kubernetes import client, config, watch

# For Kubernetes exec.
from kubernetes.stream import stream

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
# from swebench.harness.test_spec.test_spec import TestSpec
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, get_eval_type
from swebench.harness.grading import get_eval_tests_report, get_resolution_status


# SWE-Gym specific imports
from swebench_fork_swegym.harness.test_spec import make_test_spec as make_test_spec_swegym
from swebench_fork_swegym.harness.grading import get_logs_eval_new as get_logs_eval_swegym
from swebench_fork_swegym.harness.grading import get_eval_tests_report as get_eval_tests_report_swegym
from swebench_fork_swegym.harness.grading import get_resolution_status as get_resolution_status_swegym

# SWE-rebench specific imports
# from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec as make_test_spec_swerebench
# from swebench_fork_swerebench.harness.test_spec.test_spec import TestSpec
from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec as make_test_spec_swerebench
from swebench_fork_swerebench.harness.grading import get_logs_eval_new as get_logs_eval_swerebench
from swebench_fork_swerebench.harness.constants import EvalType as EvalType_SWEREBENCH
from swebench_fork_swerebench.harness.constants import FAIL_ONLY_REPOS as FAIL_ONLY_REPOS_SWEREBENCH


# SWE-rebench specific imports
from swebench_fork_swerebench.harness.grading import get_eval_tests_report as get_eval_tests_report_swerebench
from swebench_fork_swerebench.harness.grading import get_resolution_status as get_resolution_status_swerebench


# SWE-smith specific imports
from swesmith.profiles import registry as swesmith_registry
from swesmith.harness.grading import get_eval_tests_report as get_eval_tests_report_swesmith
from swesmith.constants import TEST_OUTPUT_START, TEST_OUTPUT_END



DEFAULT_NAMESPACE = "default"
DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
LSP_DOCKER_PATH = "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/condabin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

DOCKER_TLS_VERIFY = "1"
DOCKER_CERT_PATH = "pem_saved_path_including_ca.pem_cert.pem_key.pem" # Path for PEM storage; optional and can be removed if not needed.


base_dir = Path("./R2E-Gym/test_spec_cache")

def check_cache_exists(file_path) -> bool:

    print(f"check whether in: {file_path.exists()}")
    return file_path.exists()

def load_from_json(filename):
    if "swe_rebench" in str(filename):
        from swebench_fork_swerebench.harness.test_spec.test_spec import TestSpec
    elif "swe_gym" in str(filename):
        from swebench_fork_swegym.harness.test_spec import TestSpec
    elif "swe_bench" in str(filename):
        from swebench.harness.test_spec.test_spec import TestSpec
    else:
        from r2egym.agenthub.trajectory.swebench_utils import TestSpec 
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return TestSpec(**data)

##############################################################################
# Docker runtime
##############################################################################
class DockerRuntime(ExecutionEnvironment):
    """
    docker runtime is responsible for the interacting with the docker environment.
    In particular, it should allow for accomodating the features of the particualr docker envs used for r2e-edits
    - collect files
    - list files excluding test files etc
    """

    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        ip: str = None,
        logger=None,
        backend="docker",
        use_lsp : bool = False,
        **docker_kwargs,
    ):
        # check if ds is provided (required for all dockers moving forward)
        assert ds, f"Dataset not provided for docker image: {docker_image}"
        assert backend in ["docker", "kubernetes"], f"Invalid backend: {backend}"
        if logger is None:
            if self.backend == "docker":
                logger_name = "DockerRuntime"
            elif self.backend == "kubernetes":
                logger_name = "KubernetesRuntime"
            else:
                raise ValueError(f"Invalid backend: {self.backend}")
            self.logger = get_logger(logger_name)  # Pass the module name for clarity
        else:
            self.logger = logger
        self.logger.info("now in ./R2E-Gym/src/r2egym/agenthub/runtime/docker.py")
        self.use_lsp = use_lsp
        # swebench specific setup
        self.ds = ds
        self.backend = backend
        ds_image = None
        if "docker_image" in self.ds:
            ds_image = self.ds["docker_image"]
        elif "image_name" in self.ds:
            ds_image = self.ds["image_name"]
        else:
            raise ValueError(f"No docker image found in ds: {self.ds}")

        if "instance_id" in ds:
            self.instance_id = ds['instance_id']

        self.docker_image = ds_image if not docker_image else docker_image
        self.swebench_verified = "swebench" in self.docker_image or "epoch-research" in self.docker_image
        self.swesmith = "swesmith" in self.docker_image
        self.swerebench = "swerebench" in self.docker_image
        self.swegym = "xingyaoww" in self.docker_image

        if self.swesmith:
            # image_name = self.ds['image_name'].replace('__', '_1776_')
            self.swebench_verified = False
            # self.docker_image = f'jyangballin/{image_name}:latest'

        if self.swegym:
            # print(f"swegym now in docker.py, starting to make the test spec")
            self.logger.info(f"swegym now in docker.py, starting to make the test spec")

            from swebench_fork_swegym.harness.test_spec import TestSpec
            if "make_test_spec" in self.ds:
                self.logger.info("self.ds has make_test_spec, read directly")
                make_test_spec = json.loads(self.ds['make_test_spec'])
                self.test_spec = TestSpec(**make_test_spec)
                self.logger.info("has successfully read test_spec")
            
            else:
                self.cache_file = base_dir / "swe_gym" /f"{self.instance_id}.json"
                # print(f"this is dockerruntime cache_file: \n{self.cache_file}")
                self.logger.info(f"this is dockerruntime cache_file: \n{self.cache_file}")
                if check_cache_exists(self.cache_file):
                    self.test_spec = load_from_json(self.cache_file)
                    self.logger.info(f"has read cached swegym test_spec:\n{self.cache_file}")
                else:
                    self.logger.info(f"load from github")
                    self.test_spec = make_test_spec_swegym(self.ds)

            # self.logger.info(f"swegym test_spec:\n{self.test_spec}")

        if self.swebench_verified:
            # also create a test spec for swebench verified dockers (useful for grading)
            self.logger.info("started make test spec")

            from swebench.harness.test_spec.test_spec import TestSpec
            if "make_test_spec" in self.ds:
                self.logger.info("self.ds has make_test_spec, read directly")
                make_test_spec = json.loads(self.ds['make_test_spec'])
                self.test_spec = TestSpec(**make_test_spec)
                self.logger.info("has successfully read test_spec")
                
            else:
                self.cache_file = base_dir / "swe_bench_verified" /f"{self.instance_id}.json"
                self.logger.info(f"this is dockerruntime cache_file: \n{self.cache_file}")
                if check_cache_exists(self.cache_file):
                    self.test_spec = load_from_json(self.cache_file)
                    self.logger.info(f"has read cached swe-bench-verified test_spec:\n{self.cache_file}")
                else:
                    self.logger.info(f"load from github")
                    self.test_spec = make_test_spec(self.ds)

            self.logger.info("has finished make test spec")

        if self.swerebench:
            self.logger.info(f"swerebench now in docker.py, starting to make the test spec")

            from swebench_fork_swerebench.harness.test_spec.test_spec import TestSpec
            if "make_test_spec" in self.ds:
                self.logger.info("self.ds has make_test_spec, read directly")
                make_test_spec = json.loads(self.ds['make_test_spec'])
                self.test_spec = TestSpec(**make_test_spec)
                self.logger.info("has successfully read test_spec")
                
            else:
                self.cache_file = base_dir / "swe_rebench" /f"{self.instance_id}.json"
                self.logger.info(f"this is dockerruntime cache_file: \n{self.cache_file}")
                if check_cache_exists(self.cache_file):
                    self.test_spec = load_from_json(self.cache_file)
                    self.logger.info(f"has read cached swe-rebench test_spec:\n{self.cache_file}")
                else:
                    self.test_spec = make_test_spec_swerebench(self.ds)
                    self.logger.info(f"load from github")

            # self.logger.info(f"swerebench test_spec:\n{self.test_spec}")

        # set runtime params
        self.repo_path = repo_path
        self.alt_path = alt_path
        self.command = command
        self.repo_name = (
            self.ds["repo"] if self.swebench_verified or self.swesmith  or self.swegym or self.swerebench else self.ds["repo_name"]
        )
        if not self.swesmith and not self.swegym and not self.swerebench:
            self.commit_json = (
                self.ds["parsed_commit"]
                if self.swebench_verified
                else self.ds["parsed_commit_content"]
            )
            self.commit = ParsedCommit(**json.loads(self.commit_json))
        self.docker_kwargs = docker_kwargs


        self.ip = ip
        self.docker_host = r"tcp://" + self.ip + r":2375"
        custom_env = {
            'DOCKER_HOST': self.docker_host, 
            'DOCKER_TLS_VERIFY': DOCKER_TLS_VERIFY, 
            'DOCKER_CERT_PATH': DOCKER_CERT_PATH, 
            # 'DOCKER_API_VERSION': '1.40' 
        }
        print("=docker-ip="*20)
        print(f"connection to docker, use ip:{self.ip}, docker_host:{self.docker_host}")
        self.logger.info(f"connection to docker, use ip:{self.ip}, docker_host:{self.docker_host}")
        if self.backend == "docker":
            self.client = docker.from_env(timeout=120,environment=custom_env)
        elif self.backend == "kubernetes":
            # Try in-cluster config first, fallback to kubeconfig
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            self.client = client.CoreV1Api()

        # Start the container
        self.container = None
        self.container_name = self._get_container_name(self.docker_image)
        if self.backend == "kubernetes":
            # Generate a random UUID and truncate to 30 characters
            self.container_name = str(uuid.uuid4())
        self.start_container(
            self.docker_image, command, self.container_name, **docker_kwargs
        )

        # Initialize the environment
        self.setup_env()

        if self.use_lsp:
            if self.swebench_verified or self.swesmith  or self.swegym or self.swerebench:
                self.setup_lsp()
            else:
                self.uv_setup_lsp()

        if self.backend == "kubernetes":
            self.logger.info("Kubernetes environment initialized")
        else:
            print("Docker environment initialized")
            self.logger.info("Docker environment initialized")
        print("repo name: %s", self.repo_name)
        print("Docker image: %s", self.docker_image)
        self.logger.info("repo name: %s", self.repo_name)
        self.logger.info("Docker image: %s", self.docker_image)
        if self.backend == "docker":
            if self.container:
                print("Docker Container ID: %s", self.container.id)
                self.logger.info("Docker Container ID: %s", self.container.id)
        elif self.backend == "kubernetes":
            # Assuming self.container is a V1Pod object after creation/retrieval
            pod_name = (
                self.container.metadata.name
                if self.container and self.container.metadata
                else "N/A"
            )
            self.logger.info("Docker Pod Name: %s", pod_name)

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        """Return name of container"""
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-")
        image_name_sanitized = image_name_sanitized.replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def _start_kubernetes_pod(
        self, docker_image: str, command: str, pod_name: str, **docker_kwargs
    ):
        """
        Starts or connects to a Kubernetes pod with the specified configuration.

        If a pod with the given name already exists, it attempts to connect to it.
        Otherwise, it creates a new pod based on the provided image, command,
        and environment variables, then waits for it to reach the 'Running' state.

        Args:
            docker_image: The Docker image to use for the pod's container.
            command: The command to run inside the container.
            pod_name: The desired name for the Kubernetes pod.
            **docker_kwargs: Additional keyword arguments. Currently used to extract
                             'environment' variables for the pod spec.

        Raises:
            kubernetes.client.ApiException: If there's an error interacting with the
                                           Kubernetes API (other than 404 Not Found
                                           when checking existence).
            RuntimeError: If the pod fails to reach the 'Running' state after creation.
        """
        not_found_error = None
        try:
            # Check if the pod already exists
            self.container = self.client.read_namespaced_pod(
                name=pod_name, namespace=DEFAULT_NAMESPACE, _request_timeout=60,
            )
            self.logger.info(f"Found existing Kubernetes pod: {pod_name}")
            return
        except client.ApiException as e:
            not_found_error = e

        if not_found_error.status != 404:
            self.logger.error(
                f"Error checking Kubernetes pod '{pod_name}' status: {not_found_error}. Check Kubernetes configuration and permissions."
            )
            raise not_found_error

        env_vars = {"PATH": DOCKER_PATH, **docker_kwargs.get("environment", {})}
        env_spec = [{"name": k, "value": str(v)} for k, v in env_vars.items()]
        pod_body = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name},
            "spec": {
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": pod_name,
                        "image": docker_image,
                        "command": ["/bin/sh", "-c"],
                        "args": [command] if isinstance(command, str) else command,
                        "stdin": True,
                        "tty": True,
                        "env": env_spec,
                        "resources": {
                            "requests": {"cpu": "1", "memory": "1Gi"},
                        },
                    }
                ],
                "imagePullSecrets": [{"name": "dockerhub-pro"}],
                "nodeSelector": {"karpenter.sh/nodepool": "bigcpu-standby"},
                "tolerations": [
                    {
                        "key": "node.kubernetes.io/disk-pressure",
                        "operator": "Exists",
                        "effect": "NoExecute",
                        "tolerationSeconds": 10800
                    }
                ],
            },
        }

        # Create the Pod with retry logic & efficiently monitor with K8 Watch
        max_retries = 5
        backoff = 5  # seconds
        pod = None
        for attempt in range(1, max_retries + 1):
            try:
                pod = self.client.create_namespaced_pod(
                    namespace=DEFAULT_NAMESPACE, body=pod_body, _request_timeout=120,
                )
                break  # success
            except client.ApiException as e:
                # Retry on API-server throttling or transient errors
                if e.status in (409, 429, 500, 503):
                    self.logger.warning(
                        f"Transient Kubernetes error {e.status} while creating pod "
                        f"'{pod_name}' (attempt {attempt}/{max_retries}); "
                        f"retrying in {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                # Non-retryable error → propagate
                self.logger.error(f"Failed to create Kubernetes pod '{pod_name}': {e}")
                raise
        else:
            raise RuntimeError(
                f"Exceeded retry limit ({max_retries}) while creating pod '{pod_name}'."
            )

        try:
            rv = pod.metadata.resource_version
            w = watch.Watch()
            stream = w.stream(
                self.client.list_namespaced_pod,
                namespace=DEFAULT_NAMESPACE,
                field_selector=f"metadata.name={pod_name}",
                resource_version=rv,
                timeout_seconds=1200,  # 10 minutes timeout instead of 1 hour
            )
            start_time = time.time()
            for event in stream:
                obj = event["object"]
                phase = obj.status.phase
                if time.time() - start_time > 1200:
                    w.stop()
                    raise RuntimeError(f"Kubernetes pod '{pod_name}' timed out after 1200 seconds.")
                # self.logger.info(f"Event {event['type']} → pod.phase={phase}")
                if phase == "Running":
                    self.logger.info(f"Kubernetes pod '{pod_name}' is Running.")
                    w.stop()
                    break
                if phase in ["Failed", "Succeeded", "Unknown"]:
                    w.stop()
                    raise RuntimeError(
                        f"Kubernetes pod '{pod_name}' entered terminal phase '{phase}'."
                    )
            self.container = pod
        except client.ApiException as create_error:
            self.logger.error(
                f"Failed to create Kubernetes pod '{pod_name}': {create_error}"
            )
            raise create_error
        except Exception as e:
            # Handle watch timeout or other errors
            self.logger.error(f"Error waiting for pod to start: {e}")
            # Check pod status directly as fallback
            try:
                pod_status = self.client.read_namespaced_pod(
                    name=pod_name, namespace=DEFAULT_NAMESPACE, _request_timeout=60,
                )
                if pod_status.status.phase == "Running":
                    self.logger.info(f"Pod '{pod_name}' is running (verified after watch error)")
                    self.container = pod_status
                else:
                    self.logger.warning(f"Pod '{pod_name}' is in state {pod_status.status.phase}")
                    raise RuntimeError(f"Pod '{pod_name}' failed to reach Running state: {pod_status.status.phase}")
            except Exception as status_error:
                self.logger.error(f"Failed to check pod status after watch error: {status_error}")
                raise RuntimeError(f"Failed to verify pod status: {status_error}")

    def start_container(
        self, docker_image: str, command: str, ctr_name: str, **docker_kwargs
    ):
        # Start or reuse a container
        try:
            self.logger.info("begin initialing starting container")
            if self.backend == "docker":
                containers = self.client.containers.list(
                    all=True, filters={"name": ctr_name}
                )
                if containers:
                    print("has containers, use this container")
                    self.logger.info("has container, use this container")
                    self.container = containers[0]
                    if self.container.status != "running":
                        self.container.start()
                else:
                    print("no containers, start container")
                    self.logger.info("no containers, start container")
                    self.container = self.client.containers.run(
                        docker_image,
                        command=command,
                        name=ctr_name,
                        detach=True,
                        tty=True,
                        stdin_open=True,
                        network_mode='host',
                        # environment={"PATH": "/commands"},
                        **docker_kwargs,
                    )
                    print(f"this is the_started_container:{self.container}")
                    self.logger.info(f"this is the_started_container:{self.container}")
            elif self.backend == "kubernetes":
                self._start_kubernetes_pod(
                    docker_image, command, ctr_name, **docker_kwargs
                )

            self.logger.info("has finished initialied starting container")

        except Exception as e:
            self.logger.error("Container start error:", repr(e))
            self.stop_container()
            return

    def _stop_kubernetes_pod(self):
        try:
            self.client.delete_namespaced_pod(
                name=self.container_name,
                namespace=DEFAULT_NAMESPACE,
                body=kubernetes.client.V1DeleteOptions(grace_period_seconds=0),
                _request_timeout=60,
            )

            w = watch.Watch()
            stream = w.stream(
                self.client.list_namespaced_pod,
                namespace=DEFAULT_NAMESPACE,
                field_selector=f"metadata.name={self.container_name}",
                timeout_seconds=60,  # 1 minute timeout instead of indefinite
            )

            deletion_confirmed = False
            for event in stream:
                if event["type"] == "DELETED":
                    self.logger.info(f"Kubernetes pod {self.container_name} deleted.")
                    deletion_confirmed = True
                    w.stop()
                    break
            
            # If watch times out without seeing deletion, verify pod is gone
            if not deletion_confirmed:
                try:
                    # Check if pod still exists
                    self.client.read_namespaced_pod(
                        name=self.container_name, namespace=DEFAULT_NAMESPACE
                    )
                    self.logger.warning(
                        f"Watch timed out but pod {self.container_name} still exists. Forcing deletion."
                    )
                    # Try deleting again with force
                    self.client.delete_namespaced_pod(
                        name=self.container_name,
                        namespace=DEFAULT_NAMESPACE,
                        body=kubernetes.client.V1DeleteOptions(
                            grace_period_seconds=0,
                            force=True
                        ),
                    )
                except kubernetes.client.rest.ApiException as e:
                    if e.status == 404:
                        # Pod is gone, which is what we want
                        self.logger.info(f"Confirmed pod {self.container_name} is deleted.")
                    else:
                        # Some other API error
                        self.logger.error(f"Error checking pod status after timeout: {e}")
        except kubernetes.client.rest.ApiException as e:
            if e.status == 404:
                # Pod already deleted, ignore
                self.logger.info(
                    f"Kubernetes pod '{self.container_name}' not found, likely already deleted."
                )
            else:
                # Log other K8s API errors during deletion
                self.logger.error(
                    f"Error deleting Kubernetes pod '{self.container_name}': {e}"
                )
                raise e  # Re-raise unexpected errors

    def stop_container(self):
        try:
            if self.container:
                if self.backend == "docker":
                    self.container.stop()
                    self.container.remove()
                elif self.backend == "kubernetes":
                    self._stop_kubernetes_pod()
        except Exception as e:
            print("Container stop/delete error:", repr(e))

    def get_swesmith_pre_test_commands(self) -> List[str]:
        """
        Helper to construct swesmith pre-test/reset commands.
        Retrieves test files using swesmith registry and constructs a git checkout command.
        """
        rp = swesmith_registry.get_from_inst(self.ds)
        f2p_files, p2p_files = rp.get_test_files(self.ds)
        all_files = list(set(f2p_files + p2p_files))
        
        cmds = []
        if all_files:
             # Use robust command construction to avoid argument length limits if any
             cmd = (
                f'printf "%s\\n" {" ".join(all_files)} | '
                f'xargs -n1 -I{{}} git checkout HEAD~1 -- "{{}}" 2>/dev/null'
            )
             cmds.append(cmd)
        
        return cmds

    def reset_swesmith_tests(self):
        self.logger.info(f"nwo in reset_swesmith_tests")
        
        # [NEW] Use shared helper for consistency
        cmds = self.get_swesmith_pre_test_commands()
        
        self.logger.info(f"now in reset_swesmith_tests reset test files to HEAD~1 for evaluation")
        
        # Execute commands
        full_reset_command = " && ".join(cmds)
        output, error_code = self.run(full_reset_command)

        if error_code != "0":
            self.logger.error(f"Error resetting test files for swesmith: output: {output}, error_code: {error_code}")
            raise Exception(f"Error resetting test files for swesmith:  output: {output}, error_code: {error_code}")
        else:
            self.logger.info(f"successfully reset swesmith test files, output: {output}, error_code: {error_code}")


    def setup_env_swegym(self):
        try:
            self.run("git reset --hard")
            
            # Setup the run_test.sh script for subsequent testing.  
            self.logger.info(f"SWE-Gym Setting up run_tests.sh script")
            test_command = self.test_spec.eval_script
            self.logger.info(f"now in setup_env_swegym, starting to get the test command")
            self.logger.info(f"test_command:\n{test_command}")
            
            self.logger.info(f"SWE-Gym Writing eval_script_content to /run_tests.sh")
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
                temp_file.write(test_command)
                temp_file.flush()  # Ensure content is written to disk
                temp_file_path = temp_file.name
            
            self.logger.info(f"SWE-Gym Copying eval_script_content to /run_tests.sh")
            # Copy the file to container and clean up
            self.copy_to_container(temp_file_path, "/run_tests.sh")
            os.unlink(temp_file_path)  # Clean up the temporary file
            
            self.run("chmod +x /run_tests.sh")

            self.logger.info(f"SWE-Gym Linking conda env to /root/.venv")
            # Ensure can call and execute the tools in /usr/local/bin.
            self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
            self.run('echo \'export PATH="/usr/local/bin:$PATH"\' >> ~/.bashrc')
            # self.run("pip install chardet")
            self.run("pip install chardet --trusted-host pypi-mirror.weizhipin.com -i http://pypi-mirror.weizhipin.com/bzl-aliyun-pypi/simple")
            check_chardet_num_all = 3
            for check_chardet_num in range(check_chardet_num_all):
                result_check_chardet = self.run(
                            "python -m pip show chardet > /dev/null 2>&1 && echo true || echo false"
                        )
                # self.logger.warning(
                #     f"Attention!!! Is chardet installed!! {result_check_chardet}"
                # )
                if "false" in result_check_chardet[0]:
                    self.logger.info(
                    f"Re-installing chardet"
                )
                    self.run("sleep 10")
                    self.run("python -m pip install chardet")
                else:
                    break

            self.logger.info(f"Environment setup completed for swegym: {self.docker_image}")

        except Exception as e:
            self.logger.error(f"SWE-Gym Error setting up environment: {repr(e)}")

    def setup_env_swerebench(self):
        try:
            self.run("git reset --hard")
            
            # Setup the run_test.sh script for subsequent testing.  
            self.logger.info(f"SWE-rebench Setting up run_tests.sh script")
            test_command = self.test_spec.eval_script
            self.logger.info(f"now in setup_env_swerebench, starting to get the test command")
            self.logger.info(f"test_command:\n{test_command}")
            
            self.logger.info(f"SWE-rebench Writing eval_script_content to /run_tests.sh")
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
                temp_file.write(test_command)
                temp_file.flush()  # Ensure content is written to disk
                temp_file_path = temp_file.name
            
            self.logger.info(f"SWE-rebench Copying eval_script_content to /run_tests.sh")
            # Copy the file to container and clean up
            self.copy_to_container(temp_file_path, "/run_tests.sh")
            os.unlink(temp_file_path)  # Clean up the temporary file
            
            self.run("chmod +x /run_tests.sh")

            self.logger.info(f"SWE-rebench Linking conda env to /root/.venv")
            # Ensure can call and execute the tools in /usr/local/bin.
            self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
            self.run('echo \'export PATH="/usr/local/bin:$PATH"\' >> ~/.bashrc')
            # self.run("pip install chardet")
            self.run("pip install chardet --trusted-host pypi-mirror.weizhipin.com -i http://pypi-mirror.weizhipin.com/bzl-aliyun-pypi/simple")
            check_chardet_num_all = 3
            for check_chardet_num in range(check_chardet_num_all):
                result_check_chardet = self.run(
                            "python -m pip show chardet > /dev/null 2>&1 && echo true || echo false"
                        )
                # self.logger.warning(
                #     f"Attention!!! Is chardet installed!! {result_check_chardet}"
                # )
                if "false" in result_check_chardet[0]:
                    self.logger.info(
                    f"Re-installing chardet"
                )
                    self.run("sleep 10")
                    self.run("python -m pip install chardet")
                else:
                    break
            self.logger.info(f"Environment setup completed for swerebench: {self.docker_image}")

        except Exception as e:
            self.logger.error(f"SWE-rebench Error setting up environment: {repr(e)}")


    def setup_env_swesmith(self):
        try:

            # Add git mirror
            self.logger.info(f"SWE-smith Adding git mirror")
            # self.run("git config --global url.\"https://gh-proxy.com/github.com/\".insteadOf \"https://github.com/\"")

            self.run("git config --global url.\"https://hk.gh-proxy.com/https://github.com/\".insteadOf \"https://github.com/\"")


            # commit_id = self.ds['base_commit']
            self.run("git fetch")
            self.run("git status")
            self.run("git restore .")
            self.run("git reset --hard")
            # self.run(f"git checkout {commit_id}")
            self.logger.info(f"SWE-smith Checking out commit: {self.ds['instance_id']}")
            self.run(f"git checkout {self.ds['instance_id']}") # Modified: see issue: https://github.com/SWE-bench/SWE-smith/issues/118
            self.run("git clean -fdq")


            # Setup the run_test.sh script for subsequent testing.  
            self.logger.info(f"SWE-smith Setting up run_tests.sh script")

            # [NEW] Use swesmith registry to get test command
            rp = swesmith_registry.get_from_inst(self.ds)
            test_command, _ = rp.get_test_cmd(self.ds)
            
            eval_script_content = "\n".join(
                [
                    "#!/bin/bash",
                    "set -uxo pipefail",
                    "source /opt/miniconda3/bin/activate",
                    f"conda activate testbed",
                    f"cd testbed/",
                    f": '{TEST_OUTPUT_START}'", # [NEW] Use swesmith constants
                    test_command,
                    f": '{TEST_OUTPUT_END}'",   # [NEW] Use swesmith constants
                ]
            ) + "\n"
            
            self.logger.info(f"SWE-smith Writing eval_script_content to /run_tests.sh")
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
                temp_file.write(eval_script_content)
                temp_file.flush()  # Ensure content is written to disk
                temp_file_path = temp_file.name
            
            self.logger.info(f"SWE-smith Copying eval_script_content to /run_tests.sh")
            # Copy the file to container and clean up
            self.copy_to_container(temp_file_path, "/run_tests.sh")
            os.unlink(temp_file_path)  # Clean up the temporary file
            
            self.run("chmod +x /run_tests.sh")

            # self.logger.info(f"SWE-smith Linking conda env to /root/.venv")
            # Ensure can call and execute the tools in /usr/local/bin.

            bash_output, error_code = self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
            self.logger.info(f"ln -s /opt/miniconda3/envs/testbed /root/.venv:{bash_output}, error_code:{error_code}")

            self.run('echo \'export PATH="/usr/local/bin:$PATH"\' >> ~/.bashrc')
            self.run("python -m pip install chardet")
            self.logger.info(f"Environment setup completed for swesmith: {self.docker_image}")

        except Exception as e:
            self.logger.error(f"SWE-smith Error setting up environment: {repr(e)}")

    def setup_env_swebench(self):
        try:
            # make the run_tests.sh executable
            self.run("chmod +x /run_tests.sh")

            # # move all skip files (if present) to /root
            # for skip_file in SKIP_FILES:
            #     self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")
            self.alt_path = (
                "/"  # the run_test is in the "/" directory for swebench dockers
            )

            # make symlink of conda env to /root/.venv
            self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")

            # install required packages TODO: check if working
            # self.run(
            #     "python -m pip install tree-sitter==0.20.4 tree_sitter_languages==1.10.2"
            # )
            self.run("pip install chardet --trusted-host pypi-mirror.weizhipin.com -i http://pypi-mirror.weizhipin.com/bzl-aliyun-pypi/simple")

            check_chardet_num_all = 3
            for check_chardet_num in range(check_chardet_num_all):
                result_check_chardet = self.run(
                            "python -m pip show chardet > /dev/null 2>&1 && echo true || echo false"
                        )
                if "false" in result_check_chardet[0]:
                    self.logger.info(
                     f"Re-installing chardet"
                )
                    self.run("sleep 10")
                    self.run("python -m pip install chardet")
                else:
                    break

            # sudo apt-get install patchutils
            # self.run("apt-get update")
            # self.run("apt-get install -y patchutils")
        except Exception as e:
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )

    def setup_env(self):
        if self.swebench_verified:
            return self.setup_env_swebench()
        elif self.swesmith:
            self.logger.info(f"Setting up environment for swesmith: {self.docker_image}")
            return self.setup_env_swesmith()
        elif self.swegym:
            self.logger.info(f"Setting up environment for swegym: {self.docker_image}")
            return self.setup_env_swegym()
        elif self.swerebench:
            self.logger.info(f"Setting up environment for swerebench: {self.docker_image}")
            return self.setup_env_swerebench()


        self.logger.info(f"Setting up environment for r2e: {self.docker_image}")

        try:
            # setup venv
            # modify the repo path to a common path
            # self.run(f"cp -r {self.repo_path} /workspace")

            # create a symlink from repo_path/.venv to /root/.venv
            self.run(f"ln -s {self.repo_path}/.venv {self.alt_path}/.venv")

            self.run(
                f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python"
            )
            self.run(
                f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3"
            )
            self.run(
                f"find {self.repo_path}/.venv/bin -type f -executable -exec ln -sf {{}} {self.alt_path}/.local/bin/ \\;"
            )
            # print(self.run(f"ls -l {self.alt_path}/.local/bin"))

            # self.run(f"mv {self.repo_path} /workspace")
            # self.repo_path = "/workspace"

            # install required packages
            # self.run("uv pip install tree_sitter_languages") # remove since already installed in new dockers

            self.run("uv pip install chardet")

            self.run("find . -name '*.pyc' -delete")

            self.run("find . -name '__pycache__' -exec rm -rf {} +")

            # also delete pycache and pyc from /r2e_tests
            self.run("find /r2e_tests -name '*.pyc' -delete")
            self.run("find /r2e_tests -name '__pycache__' -exec rm -rf {} +")
            # move all skip files (if present) to /root
            print(SKIP_FILES_NEW)
            print(f"{self.repo_path}") #/testbed
            print(f"{self.alt_path}") #/root

            #SKIP_FILES_NEW: ['run_tests.sh', 'r2e_tests']
            for skip_file in SKIP_FILES_NEW:
                result_tuple = self.run(
                    f"[ -e {self.repo_path}/{skip_file} ] && echo true || echo false"
                )
                # [ -e testbed/run_tests.sh ] && echo true || echo false
                if "true" in result_tuple[0]:
                    self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")
                else:
                    pass
               
            
            # r2e_tests are in the / directory, move them to /root
            self.run(f"mv /r2e_tests {self.alt_path}/r2e_tests")

            # make a softlink for /root/r2e_tests (if present)
            self.run(f"ln -s {self.alt_path}/r2e_tests {self.repo_path}/r2e_tests")
            result_tuple = self.run(
                    f"[ -e /testbed/run_tests.sh ] && echo true || echo false"
                )
            print(result_tuple)

            result_tuple = self.run(
                    f"[ -e /root/r2e_tests ] && echo true || echo false"
                )
            print(result_tuple)


            result_tuple = self.run(
                    f"[ -e /root/run_tests.sh ] && echo true || echo false"
                )
            print(result_tuple)


            # exit()
            # self.run(f"ln -s /r2e_tests {self.repo_path}/r2e_tests")
        except Exception as e:
            self.logger.error(f"Error setting up environment: {repr(e)}")

    def uv_setup_lsp(self):
        bash_output, error_code = self.run("curl -fsSL https://deb.nodesource.com/setup_18.x | bash", timeout=1800)
        self.logger.info(f"run for curl -fsSL https://deb.nodesource.com/setup_18.x | bash output:{bash_output}, error_code:{error_code}")

        bash_output, error_code = self.run("apt-get install -y nodejs",timeout=600)
        self.logger.info(f"run for apt-get install -y nodejs output:{bash_output}, error_code:{error_code}")

        bash_output, error_code = self.run("uv pip install orjson -i https://pypi.tuna.tsinghua.edu.cn/simple")
        self.logger.info(f"run for pip install orjson output:{bash_output}, error_code:{error_code}")
        
        bash_output, error_code = self.run("uv pip install pyright -i https://pypi.tuna.tsinghua.edu.cn/simple")
        self.logger.info(f"run for pip install pyright output:{bash_output}, error_code:{error_code}")

        bash_output, error_code = self.run("uv pip install idna -i https://pypi.tuna.tsinghua.edu.cn/simple")
        self.logger.info(f"run for pip install idna output:{bash_output}, error_code:{error_code}")


    def setup_lsp(self):
        bash_output, error_code = self.run("curl -fsSL https://deb.nodesource.com/setup_18.x | bash", timeout=1800)
        self.logger.info(f"run for curl -fsSL https://deb.nodesource.com/setup_18.x | bash output:{bash_output}, error_code:{error_code}")

        bash_output, error_code = self.run("apt-get install -y nodejs",timeout=600)
        self.logger.info(f"run for apt-get install -y nodejs output:{bash_output}, error_code:{error_code}")

        bash_output, error_code = self.run("pip install orjson -i https://pypi.tuna.tsinghua.edu.cn/simple")
        self.logger.info(f"run for pip install orjson output:{bash_output}, error_code:{error_code}")
        
        bash_output, error_code = self.run("pip install pyright -i https://pypi.tuna.tsinghua.edu.cn/simple")
        self.logger.info(f"run for pip install pyright output:{bash_output}, error_code:{error_code}")

        bash_output, error_code = self.run("pip install idna -i https://pypi.tuna.tsinghua.edu.cn/simple")
        self.logger.info(f"run for pip install idna output:{bash_output}, error_code:{error_code}")


    def get_task_instruction(self) -> str:
        # try getting the content inside of [ISSUE] [/ISSUE] using regex tags for ds['problem_statement'] else return ds['problem_statement']
        try:
            content = self.ds["problem_statement"]
            return re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL).group(1)
        except Exception as e:
            return self.ds["problem_statement"]

    def _run_kubernetes(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir: str = "",
    ) -> tuple[str, str]:
        """
        Kubernetes-specific method to execute code or commands in the pod, with a timeout.
        Mirrors the logic of the original Docker `run` method using Kubernetes API.
        """
        # Command includes 'timeout' and potentially 'cd <workdir> &&' from the main run method
        command = ""
        if workdir:
            # Use '&&' so that failure to change directory aborts the command
            command += f"cd {workdir} && "
        command += f"timeout {timeout} {code} {args}"
        full_command = ["/bin/sh", "-c", command]
        try:
            # Define the exec function call within a lambda for the executor
            def execute_command():
                resp = stream(
                    self.client.connect_get_namespaced_pod_exec,
                    self.container_name,
                    DEFAULT_NAMESPACE,
                    command=full_command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,  # Match docker exec_run settings
                    _preload_content=False,  # Important for streaming
                )
                # Read until the command exits, accumulating each channel
                combined_chunks = []
                stdout_chunks = []
                stderr_chunks = []
                while resp.is_open():
                    resp.update(timeout=1)  # wait for data
                    if resp.peek_stdout():
                        chunk = resp.read_stdout()
                        stdout_chunks.append(chunk)
                        combined_chunks.append(chunk)
                    if resp.peek_stderr():
                        chunk = resp.read_stderr()
                        stderr_chunks.append(chunk)
                        combined_chunks.append(chunk)
                resp.close()
                exit_code = resp.returncode
                combined_output = "".join(combined_chunks)
                return combined_output, exit_code

            # Execute with an overall timeout slightly larger than the command's timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(execute_command)
                # Use timeout+10 as a buffer for k8s comms
                combined_output, exit_code = future.result(timeout=timeout + 5)

            # Process results - combined_output already preserves inter-leaved stdout/stderr
            output = combined_output

            if exit_code is None:  # Should not happen if command finished
                self.logger.error("Kubernetes exec: Exit code not found.")
                return output, "-1"  # Unknown error state

            if exit_code == 124:
                self.logger.error(f"Internal Timeout via 'timeout' command: {timeout}s")
                return f"The command took too long to execute (>{timeout}s)", "-1"

            if exit_code != 0:
                # Log format matches the docker version's error logging
                self.logger.error(
                    f"Kubernetes exec Error: Exit code {exit_code}\nError Message: {output}"
                )
                # Return combined output and error code string
                return output, f"Error: Exit code {exit_code}"

            # Remove ANSI escape codes and \r characters from the combined output
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(exit_code)
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Kubernetes exec Overall Timeout: {timeout + 5}s")
            return f"The command took too long to execute (>{timeout}s)", "-1"
        except client.ApiException as e:
            self.logger.error(f"Kubernetes API Error during exec: {e}")
            return f"Error executing command in pod: {repr(e)}", "-1"
        except Exception as e:
            self.logger.error(f"Unexpected error during Kubernetes exec: {repr(e)}")
            return f"Error: {repr(e)}", "-1"

    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir=None,
        type: str = None,
    ) -> tuple[str, str]:
        """
        General method to execute code or commands in the container, with a timeout.

        :param code: The code or command to execute.
        :param args: Arguments to pass to the code/script.
        :param workdir: The working directory inside the container (optional).
        :return: A tuple containing (output, error_message). If no error, error_message is the exit code (str).
        """
        exec_code = code
        exec_workdir = self.repo_path if workdir is None else workdir

        if self.backend == "kubernetes":
            return self._run_kubernetes(exec_code, timeout, args, workdir=exec_workdir)

        command = f"timeout {timeout} {exec_code} {args}"
        self.logger.info(f"started submit command: {command}")
        try:
            if "lsp_daemon >" in command:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    # Notice we do NOT set tty=True here
                    future = executor.submit(
                        self.container.exec_run,
                        cmd=["/bin/sh", "-c", command],
                        # cmd=command,
                        workdir=exec_workdir,
                        stdout=True,
                        stderr=True,
                        environment={
                            # "PATH": LSP_DOCKER_PATH,
                            "PATH": DOCKER_PATH,
                            # "LSP_COMMAND": "/opt/miniconda3/envs/testbed/bin/pyright-langserver --stdio",
                        },
                    )
                    exec_result = future.result(timeout=timeout + 5)      
                    # Retrieve output and exit code
                    output = exec_result.output.decode("utf-8", errors="replace")
                    error_code = exec_result.exit_code

            elif "bash /root/setup_env.sh" in command:
                self.logger.info(f"workdir: root")
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    # Notice we do NOT set tty=True here
                    future = executor.submit(
                        self.container.exec_run,
                        cmd=["/bin/sh", "-c", command],
                        # cmd=command,
                        workdir="/root",
                        stdout=True,
                        stderr=True,
                        environment={
                            "PATH": DOCKER_PATH,
                        },
                    )
                    exec_result = future.result(timeout=timeout + 5)      
                    # Retrieve output and exit code
                    output = exec_result.output.decode("utf-8", errors="replace")
                    error_code = exec_result.exit_code

          
            else:
                self.logger.info(f"workdir: {exec_workdir}")
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    # Notice we do NOT set tty=True here
                    future = executor.submit(
                        self.container.exec_run,
                        cmd=["/bin/sh", "-c", command],
                        # cmd=command,
                        workdir=exec_workdir,
                        stdout=True,
                        stderr=True,
                        environment={
                            "PATH": DOCKER_PATH,
                        },
                    )
                    exec_result = future.result(timeout=timeout + 5)      
                    # Retrieve output and exit code
                    output = exec_result.output.decode("utf-8", errors="replace")
                    error_code = exec_result.exit_code

            self.logger.info(f"fininsh execute command: {command}")

            if error_code == 124:
                self.logger.error(f"Internal Timeout: {timeout}s")
                return f"The command took too long to execute (>{timeout}s)", "-1"

            if error_code != 0:
                self.logger.error(
                    f"Error: Exit code {error_code} \nError Message: {output}"
                )
                return output, f"Error: Exit code {error_code}"

            # Remove ANSI escape codes and \r characters
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(error_code)

        ## timeout
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Timeout: {timeout}s for command: {command}")
            return f"The command took too long to execute (>{timeout}s)", "-1"

        except Exception as e:
            return f"Error: {repr(e)} for command: {command}", "-1"

    def demux_run(
        self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir=None
    ) -> tuple[str, str]:
        command = f"timeout {timeout} {code} {args}"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Set demux=True to get separate stdout and stderr streams
                future = executor.submit(
                    self.container.exec_run,
                    cmd=command,
                    workdir=self.repo_path if workdir is None else workdir,
                    demux=True,  # This is the key change
                    environment={"PATH": DOCKER_PATH},
                )
                exec_result = future.result(timeout=timeout + 5)

            # Unpack the result - when demux=True, output is a tuple of (stdout_data, stderr_data)
            output_data, error_data = exec_result.output
            error_code = exec_result.exit_code

            # Handle None cases and decode the outputs
            stdout = (
                output_data.decode("utf-8", errors="replace") if output_data else ""
            )
            stderr = error_data.decode("utf-8", errors="replace") if error_data else ""

            if error_code != 0:
                self.logger.error(
                    f"Error: Exit code {error_code} \nStdout Message: {stdout}, \nError Message: {stderr}"
                )
                return stdout, stderr, f"Error: Exit code {error_code}"

            return stdout, stderr, str(error_code)
        except Exception as e:
            return f"Error: {repr(e)}", f"Error: {repr(e)}", "-1"

    def _copy_to_container_kubernetes(self, src_path: str, dest_path: str):
        """
        Copy a file or directory from host into Kubernetes pod using tar over exec.
        """
        # Calculate destination directory and prepare in-memory tarball
        dest_dir = os.path.dirname(dest_path)
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src_path, arcname=os.path.basename(dest_path))
        tar_stream.seek(0)

        # Retry with exponential backoff
        max_retries = 5
        retry_delay = 5  # Initial delay in seconds
        for attempt in range(max_retries):
            try:
                # Exec into pod to untar into the destination directory
                exec_command = ["tar", "xmf", "-", "-C", dest_dir]
                resp = stream(
                    self.client.connect_get_namespaced_pod_exec,
                    self.container_name,
                    DEFAULT_NAMESPACE,
                    command=exec_command,
                    stderr=True,
                    stdin=True,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                )
                # Stream the tar binary data into the pod
                resp.write_stdin(tar_stream.read())
                resp.close()
                break  # Success, exit the retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Copy to container failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    retry_delay = min(retry_delay, 60)
                    tar_stream.seek(0)  # Reset the stream for the next attempt
                else:
                    self.logger.error(f"Copy to container failed after {max_retries} attempts: {str(e)}")
                    raise

    def copy_to_container(self, src_path: str, dest_path: str):
        """
        Copies a file or directory from the host into the container (Docker or Kubernetes).
        """
        if self.backend == "docker":
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(src_path, arcname=os.path.basename(dest_path))
            tar_stream.seek(0)
            self.container.put_archive(os.path.dirname(dest_path), tar_stream.read())
        else:
            # Kubernetes pod copy
            return self._copy_to_container_kubernetes(src_path, dest_path)

    @DeprecationWarning  # TODO: remove dependency on this method with new dockers
    def read_file(self, rel_file_path: str) -> str:
        output, _ = self.run(f"cat /{self.alt_path}/{rel_file_path}")
        return output

    def run_tests(self, timeout: int = 300) -> tuple[str, str]:
        output, error_code = self.run(f"bash {self.alt_path}/run_tests.sh", timeout=timeout)
        # Remove ANSI escape codes and \r characters
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        return output, error_code

    def demux_run_tests(self) -> tuple[str, str, str]:
        stdout, stderr, error_code = self.demux_run(
            f"bash {self.alt_path}/run_tests.sh"
        )
        # Remove ANSI escape codes and \r characters
        stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
        stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)
        return stdout, stderr, error_code

    def checkout(self, commit_hash: str) -> tuple[str, str]:
        output, error_code = self.run(f"git checkout {commit_hash}")
        return output, error_code

    def get_patch(self) -> str:
        """
        Get the diff of the current state of the repository.
        """
        # git add -A && git diff --cached
        # self.run("git add -A")
        output, _ = self.run("git add -A && git diff --cached")
        # output, _ = self.run("git diff")
        return output

    def create_file(self, file_path: str, content: str) -> tuple[str, str]:
        # create a local file with the content
        uuid_ = uuid.uuid4()
        file_path_ = f"{file_path}_{uuid_}"
        file_path__ = os.path.join("/tmp", file_path_)
        with open(file_path__, "w") as f:
            f.write(content)
        # copy the file to the container
        self.copy_to_container(file_path__, f"/testbed/{file_path_}")
        self.run(f"mv /testbed/{file_path_} /{file_path}")

    def apply_patch(self, patch: str) -> tuple[str, str]:
        # store the patch locally in a file identifiable by docker container id and timestamp
        # must contain unique patch name with both timestamp and docker image name
        uuid_ = uuid.uuid4()
        patch_path = f"{self.container_name}_{uuid_}.patch"
        patch_path = os.path.join("/tmp", patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        # copy the patch to / of the container
        self.copy_to_container(patch_path, f"/{patch_path}")
        # apply the patch
        output, error_code = self.run(f"git apply --whitespace=fix /{patch_path}")
        return output, error_code

    def reverse_patch(self, patch: str) -> tuple[str, str]:
        # store the patch locally in a file identifiable by docker container id and timestamp
        patch_path = f"{self.container_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.patch"
        patch_path = os.path.join("/tmp", patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        # copy the patch to / of the container
        self.copy_to_container(patch_path, f"/{patch_path}")
        # apply the patch
        output, error_code = self.run(f"git apply -R /{patch_path}")
        return output, error_code

    def get_logs_eval(
        # self, test_spec: TestSpec, content: str
        self, test_spec, content
    ) -> tuple[dict[str, str], bool]:
        """
        Retrieve evaluation results for a task instance from its corresponding log file

        Args:
            log_fp (str): path to log file
        Returns:
            bool: whether the patch applied successfully
            dict: status map

        modified from swebench/harness/grading.py
        """
        repo = test_spec.repo
        version = test_spec.version
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]

        # with open(log_fp) as f:
        # # TODO fix constant here
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            self.logger.error(f"Bad code found in log: {bad_codes}")
            return {}, False

        # elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
        #     # Test patch did not apply (should not happen at all)
        #     self.logger.error("Test patch did not apply")
        #     return {}, False

        # Get status map of evaluation results
        content = content.split(test_cmd)[-1]
        self.logger.info(f"using swebench log_parser for repo: {repo}")
        return log_parser(content, test_spec), True

    def parse_logs(self, log_output: str) -> dict:
        if self.swebench_verified:
            parsed_output, patch_apply_success = self.get_logs_eval(
                self.test_spec, log_output
            )
            return parsed_output

        elif self.swesmith:
            # [NEW] Logic for swesmith logs
            if APPLY_PATCH_FAIL in log_output:
                return {}
            if TESTS_TIMEOUT in log_output:
                return {}
            
            # Emulate swesmith.harness.grading.read_test_output string parsing logic
            # swesmith logic expects specific markers
            start_sep = f"+ : '{TEST_OUTPUT_START}'"
            end_sep = f"+ : '{TEST_OUTPUT_END}'"
            
            content = log_output
            if start_sep in content and end_sep in content:
                start_idx = content.find(start_sep)
                end_idx = content.find(end_sep)
                if start_idx <= end_idx:
                     content = content[start_idx:end_idx][len(start_sep):]
            
            # Use registry parser
            rp = swesmith_registry.get_from_inst(self.ds)
            return rp.log_parser(content)

        else:
            return parse_log_fn(f"{self.repo_name}")(log_output)
 
    def _calculate_reward_swesmith(self, get_test_output=False, timeout: int = 1800) -> Tuple[float, str]:
        self.logger.info(f"nwo in _calculate_reward_swesmith")
        self.reset_swesmith_tests()
        self.logger.info(f"nwo in _calculate_reward_swesmith after reset_swesmith_tests")
        output, error_msg = self.run("/run_tests.sh", timeout=timeout)
        self.logger.info(f"nwo in _calculate_reward_swesmith after run_tests.sh")
        self.logger.info(f"nwo in _calculate_reward_swesmith output: \n{output}")
        self.logger.info(f"nwo in _calculate_reward_swesmith error_msg: \n{error_msg}")
        
        # [NEW] Replaced ad-hoc parsing with swesmith grading logic
        eval_status_map = self.parse_logs(output)
        
        if not eval_status_map:
             return 0.0, output

        eval_ref = {
            FAIL_TO_PASS: self.ds.get(FAIL_TO_PASS, []),
            PASS_TO_PASS: self.ds.get(PASS_TO_PASS, []),
            FAIL_TO_FAIL: self.ds.get(FAIL_TO_FAIL, []),
            PASS_TO_FAIL: self.ds.get(PASS_TO_FAIL, []),
        }
        
        report = get_eval_tests_report_swesmith(
            eval_status_map, eval_ref
        )
        
        # Use swebench resolution status checker (compatible with structure returned by swesmith)
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        
        if success:
            return 1.0, output
        else:
            return 0.0, output


    def _calculate_reward_swebench(self, get_test_output=False, timeout: int = 300) -> float:
        # gt_test_patch = self.commit.get_patch(test_file=True,non_test_file=False)
        # self.apply_patch(gt_test_patch)
        out, _ = self.run(
            "/run_tests.sh", timeout=timeout
        )  # run the tests after applying the patch
        eval_status_map, found = self.get_logs_eval(self.test_spec, out)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
        }
        report = get_eval_tests_report(
            eval_status_map, eval_ref, eval_type=get_eval_type(self.test_spec)
        )
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        if get_test_output:
            return success, out
        return int(success)

    def _calculate_reward_swerebench(self, get_test_output=False, timeout: int = 300) -> float:
        # gt_test_patch = self.commit.get_patch(test_file=True,non_test_file=False)
        # self.apply_patch(gt_test_patch)
        out, _ = self.run(
            "/run_tests.sh", timeout=timeout
        )  # run the tests after applying the patch

        print(f"33333 {out}")
        eval_status_map, found = get_logs_eval_swerebench(self.test_spec, out)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
        }

        eval_type = EvalType.FAIL_ONLY if self.test_spec.repo in FAIL_ONLY_REPOS_SWEREBENCH else EvalType.PASS_AND_FAIL

        report = get_eval_tests_report_swerebench(
            eval_status_map, eval_ref, eval_type=eval_type
        )

        success = get_resolution_status_swerebench(report) == ResolvedStatus.FULL.value
        
        if get_test_output:
            return int(success), out
        return int(success)

    def _calculate_reward_swegym(self, get_test_output=False, timeout: int = 300) -> float:
        # gt_test_patch = self.commit.get_patch(test_file=True,non_test_file=False)
        # self.apply_patch(gt_test_patch)
        out, _ = self.run(
            "/run_tests.sh", timeout=timeout
        )  # run the tests after applying the patch

        # print(f"33333 {out}")
        eval_status_map, found = get_logs_eval_swegym(self.test_spec, out)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
        }
        report = get_eval_tests_report_swegym(
            eval_status_map, eval_ref
        )
        success = get_resolution_status_swegym(report) == ResolvedStatus.FULL.value
        if get_test_output:
            return int(success), out
        return int(success)

    def _calculate_reward_r2e(self, get_test_output=False, timeout: int = 300) -> float:
        # calculate reward based for r2e-edit dockers
        output, error_code = self.run_tests(timeout=timeout)
        # print(output)x
        parse = self.parse_logs(output)
        parse = decolor_dict_keys(parse)
        try:
            expected_json = self.ds["expected_output_json"]
        except Exception as e:
            expected_json = self.read_file("expected_test_output.json")

        expected: dict = json.loads(expected_json)
        expected = decolor_dict_keys(expected)
        parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
        expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

        # Compare
        if len(parse) != len(expected):
            reward = 0.0
        else:
            # If ANY mismatch, reward = 0.0, else = 1.0
            match = True
            for k in parse.keys():
                if not k:
                    continue
                if k not in expected:
                    match = False
                    break
                if parse[k] != expected[k]:
                    match = False
                    break
            reward = 1.0 if match else 0.0
        # If the caller wants the test output as well, return (reward, output)
        if get_test_output:
            return reward, output
        return reward

    def _calculate_reward(self, get_test_output=False, timeout: int = 300) ->  Union[float, Tuple[float, str]]:
        if self.swebench_verified:
            self.logger.info(f"SWE-bench-verified Calculating reward")
            return self._calculate_reward_swebench(get_test_output=get_test_output, timeout=timeout)
        elif self.swesmith:
            self.logger.info(f"SWE-smith Calculating reward")
            if get_test_output:
                real_reward, real_output = self._calculate_reward_swesmith(get_test_output=get_test_output, timeout=timeout)
                return real_reward, real_output
            else:
                real_reward = self._calculate_reward_swesmith(get_test_output=get_test_output, timeout=timeout)
                return real_reward
        elif self.swegym:
            self.logger.info(f"SWE-gym Calculating reward")
            # return self._calculate_reward_swegym(get_test_output=get_test_output, timeout=timeout)
            return self._calculate_reward_swegym(get_test_output=get_test_output, timeout=timeout)
        elif self.swerebench:
            self.logger.info(f"SWE-rebench Calculating reward")
            return self._calculate_reward_swerebench(get_test_output=get_test_output, timeout=timeout)
        else:
            return self._calculate_reward_r2e(get_test_output=get_test_output, timeout=timeout)

    def reset(self):
        self.stop_container()
        self.start_container(
            self.docker_image, self.command, self.container_name, **self.docker_kwargs
        )

    def close(self):
        self.stop_container()
        if self.backend == "docker":
            self.client.close()

    def run_swebv_regression(
        self, run_tests_regression: str | None = None, timeout: int = 300
    ) -> dict[str, str]:
        # run the regression tests for swebench verified dockers
        # copy the 'run_tests_regression' thing from ds into the container at /run_tests_regression.sh
        if run_tests_regression is None:
            run_tests_regression = self.ds["run_tests_regression"]

        with tempfile.NamedTemporaryFile("w") as f:
            f.write(run_tests_regression)
            f.flush()
            self.copy_to_container(f.name, "/run_tests_regression.sh")
        # make the script executable
        self.run("chmod +x /run_tests_regression.sh")

        # run the regression tests
        output, error_code = self.run("/run_tests_regression.sh", timeout=timeout)
        return output
        # return swebench_parse(self.ds, output)

    def start_new_branch(self, branch_name: str = "exp") -> tuple[str, str]:
        # ## save current branch-name
        # output, error_code = self.run("git branch --show-current")
        # self.current_branch = output.strip()
        # # new branch
        # output, error_code = self.run(f"git checkout -b {branch_name}")
        # # save commit hash

        output, error_code = self.run(
            "git config --global user.email 'you@example.com'"
        )
        output, error_code = self.run("git config --global user.name 'Your Name'")
        output, error_code = self.run("git rev-parse HEAD")
        self.current_commit = output.strip()
        return output, error_code

    def commit_after_step(self, step_idx: int) -> tuple[str, str]:
        # commit
        output, error_code = self.run("git add .")
        output, error_code = self.run(f"git commit -m '{step_idx}'")
        return output, error_code

    def undo_last_commit(self) -> tuple[str, str]:
        # undo last commit
        output, error_code = self.run("git reset --hard HEAD~1")
        return output, error_code

    def get_current_commit_hash(self) -> str:
        output, _ = self.run("git rev-parse HEAD")
        return output.strip()

    def soft_git_reset(self) -> tuple[str, str]:
        # soft reset to saved commit
        output, error_code = self.run(f"git reset --soft {self.current_commit}")

        # # checkout to saved branch
        # output, error_code = self.run(f"git checkout {self.current_branch}")

        return output, error_code
