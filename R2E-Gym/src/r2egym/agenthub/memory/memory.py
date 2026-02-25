import os
import json
import time
import traceback
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import litellm
import re
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.trajectory import TrajectoryStep

logger = get_logger(__name__)

MAX_SUMMARY_LENGTH = 300000

def extract_content_after_think(text):
    # Use regular expressions to match the <think> block and all preceding content.
    # re.DOTALL allows the dot (.) to match newline characters.
    # This greedy pattern (.*) finds the last occurrence of </think> and captures all content following it.
    pattern = r".*</think>\s*(.*)"
    
    match = re.search(pattern, text, flags=re.DOTALL)
    
    if match:
        # Return the captured group (content after </think>), trimmed of leading/trailing whitespace.
        return match.group(1).strip()
    
    # If the tag is not found, return the original string.
    return text

@dataclass
class SingleTurnSummary:
    """Single turn summary data structure"""
    step_idx: int
    signal_type: str  # [CRITICAL], [USEFUL], [EXPLORATORY], [NOISE]
    action_summary: str
    observation_summary: str
    key_info: str
    raw_content: str
    action: str
    observation: str
    timestamp: float = 0.0
    token_usage_prompt: int = 0
    token_usage_completion: int = 0
    token_usage_total: int = 0


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleTurnSummary":
        return cls(**data)


@dataclass
class AggregatedSummary:
    """Aggregated summary for a window of turns"""
    start_idx: int
    end_idx: int
    summary: str
    timestamp: float = 0.0
    token_usage_prompt: int = 0
    token_usage_completion: int = 0
    token_usage_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedSummary":
        return cls(**data)


class Memory:
    """
    Memory manager for agent conversation history with compression capabilities.
    
    This class handles:
    - Single-turn summarization
    - Multi-turn aggregation
    - History compression with sliding window
    - Memory statistics and export
    """
    
    def __init__(
        self,
        single_turn_summary_prompt: str,
        multi_turns_summary_prompt: str,
        problem_statement: str,
        model_query_fn: callable,
        logger: Optional[Any] = None,
        enable_compression: bool = True,
        summary_window: int = 10,
        keep_recent: int = 5,
        compression_trigger_step: int = 15,
        use_single_summary: bool = True,
        max_workers: int = 10
    ):
        """
        Initialize Memory manager.
        
        Args:
            single_turn_summary_prompt: Prompt template for single turn summarization
            multi_turns_summary_prompt: Prompt template for multi-turn aggregation
            model_query_fn: Function to query the LLM (should accept messages and temperature)
            logger: Logger instance
            enable_compression: Whether to enable history compression
            summary_window: Number of turns to aggregate into one summary
            keep_recent: Number of recent turns to keep in full detail
            compression_trigger_step: Step number to start compression
            use_single_summary: Whether to use single summaries or aggregated summaries
            max_workers: Maximum number of parallel workers for summarization
        """
        self.single_turn_summary_prompt = single_turn_summary_prompt
        self.multi_turns_summary_prompt = multi_turns_summary_prompt

        self.problem_statement = problem_statement
        self.model_query_fn = model_query_fn
        self.logger = logger or get_logger(self.__class__.__name__)
        
        # Configuration
        self.enable_compression = enable_compression
        self.summary_window = summary_window
        self.keep_recent = keep_recent
        self.compression_trigger_step = compression_trigger_step
        self.use_single_summary = use_single_summary
        self.max_workers = max_workers
        
        # State
        self.reset()
    
    def reset(self):
        """Reset memory state"""
        self._last_compressed_step = 0
        self._last_actual_append_history_step = 0
        self._single_summaries: List[SingleTurnSummary] = []
        self._aggregated_summaries: List[AggregatedSummary] = []
        self._compression_history: List[Dict[str, Any]] = []
    
    def _parse_single_turn_summary(self, summary_content: str, step_idx: int, raw_action:str, observation:str, token_infos:dict) -> SingleTurnSummary:
        """
        Parse the summary response into structured format.
        
        Expected format:
        [SIGNAL_TYPE]
        Action: <summary>
        Observation: <summary>
        Key Info: 
        - <item1>
        - <item2>
        """
        import re
        
        # Extract signal type
        signal_match = re.search(r'\[(CRITICAL|USEFUL|EXPLORATORY|NOISE)\]', summary_content)
        signal_type = signal_match.group(0) if signal_match else "[NOISE]"
        
        # Extract action summary
        action_match = re.search(r'Action:\s*(.+?)(?=\n(?:Observation:|Key Info:|$))', summary_content, re.DOTALL)
        action_summary = action_match.group(1).strip() if action_match else ""
        
        # Extract observation summary
        obs_match = re.search(r'Observation:\s*(.+?)(?=\n(?:Key Info:|$))', summary_content, re.DOTALL)
        observation_summary = obs_match.group(1).strip() if obs_match else ""
        
        # Extract key info
        key_info_match = re.search(r'Key Info:\s*(.+?)$', summary_content, re.DOTALL)
        key_info = key_info_match.group(1).strip() if key_info_match else ""

        return SingleTurnSummary(
            step_idx=step_idx,
            signal_type=signal_type,
            action_summary=action_summary,
            observation_summary=observation_summary,
            key_info=key_info,
            raw_content=summary_content,
            timestamp=time.time(),
            action=raw_action,
            observation=observation,
            token_usage_prompt=token_infos['token_usage_prompt'],
            token_usage_completion=token_infos['token_usage_completion'],
            token_usage_total=token_infos['token_usage_total'],
        )
    
    def single_turn_summary(
        self, 
        thought: str, 
        action_str: str, 
        observation: str, 
        step_idx: int, 
    ) -> SingleTurnSummary:
        """
        Summarize a single turn (action-observation pair).
        
        Args:
            thought: The thought/reasoning from the agent
            action_str: The action string (XML format)
            observation: The observation/result from environment
            step_idx: Current step index
            
        Returns:
            SingleTurnSummary object containing the summary
        """
        if self.use_single_summary:
            try:
                raw_action = f"{thought}\n{action_str}"
                # Prepare the prompt
                single_turn_prompt = self.single_turn_summary_prompt.format(
                    action=f"{thought}\n{action_str}",
                    observation=observation
                )
                
                # Query the model for summary
                summary_messages = [
                    {"role": "user", "content": single_turn_prompt}
                ]
                
                summary_response, llm_exec_time = self.model_query_fn(
                    messages=summary_messages,
                    temperature=0.7
                )

                #  # ---------- Statistics: tokens, cost, and execution time -----------
                if hasattr(summary_response, "usage"):
                    usage = summary_response.usage
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)
                    total_tokens = getattr(usage, "total_tokens", 0)

                    token_infos = {
                        "token_usage_prompt":prompt_tokens,
                        "token_usage_completion":completion_tokens,
                        "token_usage_total":total_tokens,
                    }
                    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                    self.logger.warning(f"Prompt Token Details: {prompt_tokens_details}")
                    self.logger.info(
                        f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                    )
                else:
                    usage = 0
                    completion_tokens = 0
                    prompt_tokens = 0
                    total_tokens = 0
                    # total_tokens =  self._count_tokens(messages)
                    token_infos = {
                        "token_usage_prompt":prompt_tokens,
                        "token_usage_completion":completion_tokens,
                        "token_usage_total":total_tokens,
                    }
                    self.logger.warning(
                        "No token usage information available in the response."
                    )
                # ---------- end -----------

                summary_content = extract_content_after_think(summary_response.choices[0].message.content)
                
                # Parse the summary response
                summary = self._parse_single_turn_summary(summary_content, step_idx, raw_action, observation, token_infos)
                
                self.logger.info(f"[Step {step_idx}] Single turn type: {summary.signal_type}")
                if not summary_content:
                    print(11111)
                self.logger.warning(f"[Step {step_idx}] Single turn summary: {summary_content}")
                
                return summary
                
            except Exception as e:
                self.logger.error(f"Error in single_turn_summary at step {step_idx}: {e}")
                # Fallback: return a basic summary
                return SingleTurnSummary(
                    step_idx=step_idx,
                    signal_type="[FAILED]",
                    action_summary=f"Step {step_idx}: Executed action",
                    observation_summary="Summary generation failed",
                    key_info="",
                    raw_content=f"Failed: {str(e)}",
                    timestamp=time.time(),
                    action=raw_action,
                    observation=observation,
                )

        else:
            raw_action = f"{thought}\n{action_str}"

            return SingleTurnSummary(
                step_idx=step_idx,
                signal_type="",
                action_summary=f"Not use single summary",
                observation_summary="",
                key_info="",
                raw_content="",
                timestamp=time.time(),
                action=raw_action,
                observation=observation,
                token_usage_prompt=0,
                token_usage_completion=0,
                token_usage_total=0,
            )

    def multi_turns_summary(
        self, 
        start_idx: int, 
        end_idx: int, 
        single_summaries: List[SingleTurnSummary], 
        problem_statement: str
    ) -> AggregatedSummary:
        """
        Aggregate multiple single-turn summaries into a comprehensive summary.
        
        Args:
            start_idx: Starting step index (inclusive)
            end_idx: Ending step index (inclusive)
            single_summaries: List of SingleTurnSummary objects
            problem_statement: The original issue/problem statement
            
        Returns:
            AggregatedSummary object
        """
        try:
            # Format single summaries for the prompt
            formatted_summaries = ""
            for i, summary in enumerate(single_summaries):
                step_num = start_idx + i
                formatted_summaries += f"\n### Turn {step_num}\n"
                if self.use_single_summary:
                    formatted_summaries += f"{summary.signal_type}\n"
                    formatted_summaries += f"Model Action: {summary.action}\n"
                    formatted_summaries += f"Environment Observation: {summary.observation}\n"
                    formatted_summaries += f"Summary Action: {summary.action_summary}\n"
                    formatted_summaries += f"Summary Observation: {summary.observation_summary}\n"
                    if summary.key_info:
                        formatted_summaries += f"Key Info: {summary.key_info}\n"
                else:
                    formatted_summaries += f"Model Action: {summary.action}\n"
                    formatted_summaries += f"Environment Observation: {summary.observation}\n"
                formatted_summaries += "-" * 40 + "\n"
            
            if len(formatted_summaries) > MAX_SUMMARY_LENGTH:
                formatted_summaries = formatted_summaries[:MAX_SUMMARY_LENGTH] + "!!!Note that content is truncated because it's too long!"

            # Prepare the multi-turn summary prompt
            multi_turn_prompt = self.multi_turns_summary_prompt.format(
                summary_window=self.summary_window,
                issue_description=problem_statement,
                start_turn=start_idx,
                end_turn=end_idx,
                turn_summaries=formatted_summaries
            )
            
            # Query the model for aggregated summary
            summary_messages = [
                {"role": "user", "content": multi_turn_prompt}
            ]
            
            summary_response, llm_exec_time = self.model_query_fn(
                messages=summary_messages,
                temperature=0.7
            )

            #  # ---------- Statistics: tokens, cost, and execution time -----------
            if hasattr(summary_response, "usage"):
                usage = summary_response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)
                token_infos = {
                    "token_usage_prompt":prompt_tokens,
                    "token_usage_completion":completion_tokens,
                    "token_usage_total":total_tokens,
                }
                prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                self.logger.warning(f"Prompt Token Details: {prompt_tokens_details}")
                self.logger.info(
                    f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                )
            else:
                completion_tokens = -1
                prompt_tokens = -1
                total_tokens = -1
                token_infos = {
                    "token_usage_prompt":prompt_tokens,
                    "token_usage_completion":completion_tokens,
                    "token_usage_total":total_tokens,
                }
                self.logger.warning(
                    "No token usage information available in the response."
                )
            # ---------- end -----------

            
            aggregated_content = "<Memory>" + extract_content_after_think(summary_response.choices[0].message.content) + "</Memory>"
            # aggregated_content = extract_content_after_think(summary_response.choices[0].message.content)
            
            self.logger.info(f"Multi-turn summary generated for turns {start_idx}-{end_idx}")
            
            return AggregatedSummary(
                start_idx=start_idx,
                end_idx=end_idx,
                summary=aggregated_content,
                timestamp=time.time(),
                token_usage_prompt=token_infos['token_usage_prompt'],
                token_usage_completion=token_infos['token_usage_completion'],
                token_usage_total=token_infos['token_usage_total'],
            )
            
        except Exception as e:
            self.logger.error(f"Error in multi_turns_summary for turns {start_idx}-{end_idx}: {e}")
            # Fallback: return a basic concatenation
            fallback = f"## Turns {start_idx}-{end_idx} Summary\n\n"
            fallback += "Summary generation failed. Key steps:\n"
            for i, summary in enumerate(single_summaries):
                if summary.signal_type in ['[CRITICAL]', '[USEFUL]']:
                    fallback += f"- Step {start_idx + i}: {summary.action_summary}\n"
            
            return AggregatedSummary(
                start_idx=start_idx,
                end_idx=end_idx,
                summary=fallback,
                timestamp=time.time()
            )
    
    def compress_history(
        self, 
        current_step: int,
        trajectory_steps: List[TrajectoryStep],
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Compress conversation history by replacing old detailed turns with summaries.
        
        Args:
            current_step: Current step number
            trajectory_steps: List of TrajectoryStep objects
            history: Current conversation history
            
        Returns:
            Compressed history
        """
        # Check if compression is needed
        if not self.enable_compression:
            return history
        
        if current_step < self.compression_trigger_step:
            return history

        # Only compress if we have enough steps
        if current_step < self.summary_window + self.keep_recent:
            return history
        
        # Calculate which turns need to be summarized
        compress_until = current_step - self.keep_recent


        # If already compressed recently or do not reach the compressed turns, skip
        if compress_until < self._last_actual_append_history_step + self.summary_window:
            return history
        
        try:
            compression_start_time = time.time()
            
            # Step 1: Generate single-turn summaries in parallel
            futures = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # From last compressed step to current compress point; processed single_summaries are recorded here
                for step_idx in range(self._last_compressed_step, compress_until):
                    if step_idx < len(trajectory_steps):
                        step = trajectory_steps[step_idx]
                        
                        # Submit task to thread pool
                        future = executor.submit(
                            self.single_turn_summary,
                            thought=step.thought,
                            action_str=step.action,
                            observation=step.observation,
                            step_idx=step_idx,
                        )
                        futures.append((step_idx, future))
                
                # Collect results in order
                sorted_futures = sorted(futures, key=lambda item: item[0])
                
                new_summaries = []
                for step_idx, future in sorted_futures:
                    try:
                        single_summary = future.result()
                        new_summaries.append(single_summary)
                    except Exception as exc:
                        self.logger.error(f'Step {step_idx} summary generation failed: {exc}')
                
                # Append new summaries
                self._single_summaries.extend(new_summaries)
            
            # Step 2: Reconstruct history with compression
            # Keep system and initial user prompt

            # 20 Trigger, summary 15, 25 Trigger 
            last_complete_windows = self._last_actual_append_history_step // self.summary_window  # Windows last added to history
            new_history = history[:2 + last_complete_windows].copy()
            num_complete_windows = len(self._single_summaries) // self.summary_window    # Total windows
            remaining_windows = len(self._single_summaries) % self.summary_window

            for window_idx in range(last_complete_windows, num_complete_windows):
                start_idx = window_idx * self.summary_window #   window_idx * 10, window_idx (0, self._singe_summaries/10)
                end_idx = start_idx + self.summary_window - 1
                
                window_summaries = self._single_summaries[start_idx:end_idx + 1]
                
                if self.use_single_summary:

                    single_summary_aggs = "" 
                    for single_summary in window_summaries:
                        formatted = f"Turn {single_summary.step_idx}\n{single_summary.signal_type}\n"
                        formatted += f"Action: {single_summary.action_summary}\n"
                        formatted += f"Observation: {single_summary.observation_summary}\n"
                        if single_summary.key_info:
                            formatted += f"Key Info: {single_summary.key_info} \n"
                        single_summary_aggs += formatted
                    
                    new_history.append({
                        "role": "user",
                        "content": single_summary_aggs
                    })

                # Generate aggregated summaries for complete windows
                else:
                    # Use aggregated summaries
                    # Get problem statement from history
                    # problem_statement = history[1]['content'] if len(history) > 1 else ""
                    
                    aggregated = self.multi_turns_summary(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        single_summaries=window_summaries,
                        problem_statement=self.problem_statement
                    )
                    
                    self._aggregated_summaries.append(aggregated)


                    new_history.append({
                        'role': 'user',
                        'content': f"[COMPRESSED HISTORY - Turns {aggregated.start_idx}-{aggregated.end_idx}]\n{aggregated.summary}"
                    })
            
            # # Clean up aggregated single summaries
            # num_summarized = num_complete_windows * self.summary_window
            # self._single_summaries = self._single_summaries[num_summarized:]
            
            # Step 3: Add recent detailed turns
            messages_per_turn = 2
            keep_messages = (self.keep_recent + remaining_windows) * messages_per_turn
            
            if len(history) > keep_messages + 2:
                new_history.extend(history[-(keep_messages):])
            else:
                new_history.extend(history[2:])
            
            # Update state
            self._last_compressed_step = compress_until
            self._last_actual_append_history_step = compress_until - remaining_windows
            # Record compression event
            compression_time = time.time() - compression_start_time
            self._compression_history.append({
                'step': current_step,
                'compressed_until': compress_until,
                'num_single_summaries': len(self._single_summaries),
                'num_aggregated_summaries': len(self._aggregated_summaries),
                'new_history_length': len(new_history),
                'compression_time': compression_time,
                'timestamp': time.time()
            })
            
            self.logger.info(
                f"Compressed history up to step {compress_until}. "
                f"New history length: {len(new_history)} messages. "
                f"Time: {compression_time:.2f}s"
            )
            
            return new_history
            
        except Exception as e:
            self.logger.error(f"Error in compress_history: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # On error, return original history
            return history
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory compression.
        
        Returns:
            Dictionary with compression statistics
        """
        return {
            'compressed': self._last_compressed_step > 0,
            'last_compressed_step': self._last_compressed_step,
            'num_single_summaries': len(self._single_summaries),
            'num_aggregated_summaries': len(self._aggregated_summaries),
            'compression_events': len(self._compression_history),
            'total_compressions': sum(1 for _ in self._compression_history),
            'config': {
                'enable_compression': self.enable_compression,
                'summary_window': self.summary_window,
                'keep_recent': self.keep_recent,
                'compression_trigger_step': self.compression_trigger_step,
                'use_single_summary': self.use_single_summary,
                'max_workers': self.max_workers
            }
        }
    
    def export_memory(self, history: List[Dict[str, str]], instance_id:str, output_path: Path):
        """
        Export complete memory state to a JSON file for inspection.
        
        Args:
            output_path: Path to save the memory export
        """
        try:
            memory_export = {
                'instance_id': instance_id,
                'config': {
                    'enable_compression': self.enable_compression,
                    'summary_window': self.summary_window,
                    'keep_recent': self.keep_recent,
                    'compression_trigger_step': self.compression_trigger_step,
                    'use_single_summary': self.use_single_summary,
                    'max_workers': self.max_workers
                },
                'state': {
                    'last_compressed_step': self._last_compressed_step,
                    'num_single_summaries': len(self._single_summaries),
                    'num_aggregated_summaries': len(self._aggregated_summaries)
                },
                'single_summaries': [s.to_dict() for s in self._single_summaries],
                'aggregated_summaries': [s.to_dict() for s in self._aggregated_summaries],
                'history': [s for s in history],
                'compression_history': self._compression_history,
                'export_timestamp': time.time()
            }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.mkdir(parents=True, exist_ok=True)

            # Create specific JSON file names under the directory
            json_filename = f"{instance_id}.json"  
            json_filepath = output_path / json_filename

            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(memory_export, f, ensure_ascii=False, indent=2) 
                
            self.logger.info(f"Memory exported to {json_filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting memory: {e}")
    
    @classmethod
    def load_memory(cls, input_path: Path) -> "Memory":
        """
        Load memory state from a JSON file.
        
        Args:
            input_path: Path to the memory export file
            
        Returns:
            Memory object with loaded state
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Create memory instance with config
        memory = cls(
            single_turn_summary_prompt="",  # Placeholder
            multi_turns_summary_prompt="",  # Placeholder
            model_query_fn=None,  # Won't be used for loaded memory
            **data['config']
        )
        
        # Restore state
        memory._last_compressed_step = data['state']['last_compressed_step']
        memory._single_summaries = [
            SingleTurnSummary.from_dict(s) for s in data['single_summaries']
        ]
        memory._aggregated_summaries = [
            AggregatedSummary.from_dict(s) for s in data['aggregated_summaries']
        ]
        memory._compression_history = data['compression_history']
        
        return memory
    
    def get_summary_report(self) -> str:
        """
        Generate a human-readable summary report.
        
        Returns:
            Formatted string report
        """
        report = "=" * 70 + "\n"
        report += "MEMORY SUMMARY REPORT\n"
        report += "=" * 70 + "\n\n"
        
        # Configuration
        report += "Configuration:\n"
        report += f"  - Compression Enabled: {self.enable_compression}\n"
        report += f"  - Summary Window: {self.summary_window} turns\n"
        report += f"  - Keep Recent: {self.keep_recent} turns\n"
        report += f"  - Trigger Step: {self.compression_trigger_step}\n"
        report += f"  - Use Single Summary: {self.use_single_summary}\n\n"
        
        # State
        report += "Current State:\n"
        report += f"  - Last Compressed Step: {self._last_compressed_step}\n"
        report += f"  - Single Summaries: {len(self._single_summaries)}\n"
        report += f"  - Aggregated Summaries: {len(self._aggregated_summaries)}\n"
        report += f"  - Compression Events: {len(self._compression_history)}\n\n"
        
        # Single summaries breakdown
        if self._single_summaries:
            report += "Single Summaries by Signal Type:\n"
            signal_counts = {}
            for s in self._single_summaries:
                signal_counts[s.signal_type] = signal_counts.get(s.signal_type, 0) + 1
            for signal, count in sorted(signal_counts.items()):
                report += f"  {signal}: {count}\n"
            report += "\n"
        
        # Compression history
        if self._compression_history:
            report += "Compression History:\n"
            for event in self._compression_history:
                report += f"  - Step {event['step']}: Compressed until {event['compressed_until']} "
                report += f"(Time: {event['compression_time']:.2f}s)\n"
            report += "\n"
        
        report += "=" * 70 + "\n"
        
        return report

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts the tokens for a list of messages using the litellm library.
        Adjust as needed depending on the model and library.
        """
        token_count = litellm.token_counter(model=self.llm_name, messages=messages)
        self.logger.info(f"Total tokens in conversation: {token_count}")
        return token_count
