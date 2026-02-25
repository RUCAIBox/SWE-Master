import copy
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class MathAgent(BaseAgent):
    """
    A math agent that solves mathematical problems step by step, following the BaseAgent interface.
    """

    def __init__(self, accumulate_thinking=True):
        """
        Initialize the MathAgent.
        """
        self.instruction = "Let's think step by step, and put your final answer within \\boxed{}."
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Process environment feedback and update internal state."""

        # Format observation based on whether it's the initial problem or subsequent feedback
        if not self.trajectory.steps:
            # Initial problem presentation
            assert isinstance(observation, dict) and "question" in observation
            question = observation["question"]
            formatted_observation = f"{question} {self.instruction}"
        else:
            # Follow-up correction prompt
            formatted_observation = "Your previous answer may contain a mistake. Please review it carefully and answer again. Put your final answer within \\boxed{}."

        self.messages.append({"role": "user", "content": formatted_observation})

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state based on the model's response.
        """
        self.messages.append({"role": "assistant", "content": response})
        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions))
        self.trajectory.steps.append(new_step)

        return Action(action=response)

    def reset(self):
        """Reset agent state for new episode."""
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        # remove thinking from assistant messages if not accumulate_thinking except the last one
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
