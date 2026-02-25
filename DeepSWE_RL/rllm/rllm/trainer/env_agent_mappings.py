def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


# Import environment classes
ENV_CLASSES = {
    "browsergym": safe_import("rllm.environments.browsergym.browsergym", "BrowserGymEnv"),
    "frozenlake": safe_import("rllm.environments.frozenlake.frozenlake", "FrozenLakeEnv"),
    "tool": safe_import("rllm.environments.tools.tool_env", "ToolEnvironment"),
    "math": safe_import("rllm.environments.base.single_turn_env", "SingleTurnEnvironment"),
    "code": safe_import("rllm.environments.base.single_turn_env", "SingleTurnEnvironment"),
    "swe": safe_import("rllm.environments.swe.swe", "SWEEnv"),
    "swe_sim": safe_import("rllm.environments.swe.swe_sim", "SWEEnv"),
    "swe_arsenal": safe_import("rllm.environments.swe.swe_arsenal","SWEEnv"),
    "competition_coding": safe_import("rllm.environments.code.competition_coding", "CompetitionCodingEnv"),
}

# Import agent classes
AGENT_CLASSES = {
    "miniwobagent": safe_import("rllm.agents.miniwob_agent", "MiniWobAgent"),
    "frozenlakeagent": safe_import("rllm.agents.frozenlake_agent", "FrozenLakeAgent"),
    "tool_agent": safe_import("rllm.agents.tool_agent", "ToolAgent"),
    "sweagent": safe_import("rllm.agents.swe_agent", "SWEAgent"),
    "math_agent": safe_import("rllm.agents.math_agent", "MathAgent"),
    "code_agent": safe_import("rllm.agents.code_agent", "CompetitionCodingAgent"),
}

# Filter out None values for unavailable imports
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}
