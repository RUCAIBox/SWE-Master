#!/usr/bin/env python3
"""
LSP Tool Evaluation Script
Using LLMs to evaluate the effectiveness of LSP tools in code problem-solving.
"""

import json
import re
import time
import litellm
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from datetime import datetime
import os

LLM_JUDGE_PROMPT="""
You are an expert code analysis assistant tasked with evaluating the effectiveness of LSP (Language Server Protocol) tools in helping AI agents solve complex software engineering problems.

This is the trajectory:{trajectory}\n

## Your Task
Analyze the provided trajectory where an AI agent attempts to solve a coding problem from the SWE-bench dataset. Your focus is to judge whether the `lsp_tool` really brings **positive contributions** to the problem-solving process or not.
If there are any errors in the return result of lsp_tool, or if a meaningless call is made (the return result does not contain useful information), please immediately output 'no' (i.e. lsp_tool is not helpful for this trajectory)
Please think carefully and maintain an objective, rational analysis. You will be punished if you are overly biased.

## Context: The lsp_tool
The `lsp_tool` is a Pyright-based code intelligence tool that provides:
- **Semantic code understanding**: Goes beyond text search by analyzing AST and symbol tables
- **Navigation capabilities**: `get_definition`, `get_type_definition`, `get_references`
- **Code structure analysis**: `get_document_symbols`, `get_workspace_symbols`
- **Call relationship analysis**: `get_call_hierarchy`, `get_incoming_calls`, `get_outgoing_calls`
- **Contextual information**: `get_hover` for docstrings and type information

This contrasts with simpler tools like `grep`, `sed`, or `cat` which only perform text-based operations.

## Evaluation Framework

### 1. Positive Indicators (Evidence that lsp_tool is helpful)
Carefully identify instances where lsp_tool:

**a) Efficient Navigation**
- Successfully locates function/class definitions across multiple files
- Finds symbol references faster than grep-based approaches
- Discovers code relationships that would be hard to find manually

**b) Accurate Code Understanding**
- Provides structural overview via `get_document_symbols` before diving into details
- Uses `get_workspace_symbols` to quickly locate relevant code entities
- Employs `get_references` to understand how a function/class is used across the codebase

**c) Strategic Analysis**
- Uses `get_call_hierarchy` to understand function dependencies
- Leverages semantic information to make informed decisions
- Follows code flow using `get_definition` chains

**d) Problem-Solving Efficiency**
- Reduces the number of file reads needed
- Avoids blind searching through irrelevant files
- Quickly identifies the right files to modify

### 2. Negative Indicators (Evidence of ineffective usage)
Also identify cases where:

**a) Tool Misuse**
- Uses lsp_tool when simple `grep` or `cat` would suffice
- Makes redundant lsp_tool calls that don't add new information
- Uses wrong commands (e.g., `get_workspace_symbols` with overly generic queries)

**b) Failed Tool Calls**
- LSP commands fail and the agent doesn't adapt
- Symbol not found errors that waste time
- Incorrect parameters leading to errors

**c) Ignored Results**
- Agent calls lsp_tool but doesn't utilize the returned information
- Makes decisions that contradict lsp_tool findings
- Proceeds with manual file exploration after lsp_tool already provided the answer

**d) Inefficiency**
- Over-reliance on lsp_tool when the problem is straightforward
- Using multiple lsp_tool calls where one would suffice
- Mixing lsp_tool with redundant text-based searches

### 3. Comparative Analysis
Compare scenarios:
- **With lsp_tool**: How does the agent locate code? How many steps?
- **Without lsp_tool (hypothetical)**: How would the agent likely proceed using only grep/cat?
- Does lsp_tool provide a **decisive advantage** in understanding code structure?

## Output Format

Provide your analysis in the following structured format:

### Executive Summary (2-3 sentences)
Provide a clear verdict: Does lsp_tool bring positive contributions? Is it helpful, neutral, or counterproductive?

### Detailed Analysis

#### Section A: Positive Contributions
If have, for each positive use case found:
```
**Use Case #X: [Brief Title]**
- **Step Number**: [which step in the trace]
- **LSP Command**: [which command was used]
- **What it found**: [summarize the output]
- **How it helped**: [explain the benefit]
- **Efficiency gain**: [compare with alternative approaches]
```

#### Section B: Ineffective/Problematic Usage
If have, for each problematic use case:
```
**Issue #X: [Brief Title]**
- **Step Number**: [which step]
- **Problem Type**: [misuse/failed call/ignored result/inefficiency]
- **What went wrong**: [describe the issue]
- **Impact**: [how it affected problem-solving]
```

#### Section C: Key Statistics
Provide quantitative measures:
- Total lsp_tool calls: X
- Successful calls: Y (Z%)
- Calls that directly contributed to solution: N
- Failed/error calls: M
- Redundant/unnecessary calls: K

#### Section D: Critical Moments
If have, identify 1-3 pivotal moments where lsp_tool either:
- **Made a breakthrough**: Enabled the agent to find the root cause
- **Caused confusion**: Led the agent in the wrong direction
- **Was underutilized**: Could have helped but wasn't used

### Overall Assessment

**Answer**: \\boxed{{yes or no}}

**Recommendations**: 
- What could be improved in how the agent uses lsp_tool?
- Are there missed opportunities where lsp_tool should have been used?
- Should any usage patterns be avoided?

## Important Guidelines

1. **Be Evidence-Based**: Every claim must reference specific step numbers and command outputs
2. **Be Objective**: Don't assume lsp_tool is good or bad; let the evidence speak
3. **Consider Context**: A failed lsp_tool call isn't inherently bad if the agent adapts well
4. **Think Counterfactually**: Would the agent have struggled more without lsp_tool?
5. **Focus on Problem-Solving**: The ultimate question is: did it help solve the actual issue?
6. **Note Tool Synergy**: Sometimes lsp_tool + grep together are better than either alone

## Special Attention Areas

- Look for cases where `get_workspace_symbols` quickly locates the right class/function
- Check if `get_document_symbols` helps understand file structure before editing
- Observe whether `get_references` reveals unexpected usage patterns
- See if `get_definition` chains help trace code flow
- Notice when the agent **should have used** lsp_tool but didn't

Begin your analysis now.
"""


def setup_logging(log_dir="llm_as_judge_logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"llm_judge_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created: {log_file}")
    
    return logger, log_file


def read_datas(file_path):
    with open(file_path, "r") as f:
        datas = [json.loads(line) for line in f]
    return datas


def rule_filter_datas(datas, logger, limit_lsp: int = 1):
    """Filter data based on rules using regex"""
    function_block_regex = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
    parameter_regex = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

    filtered_datas = []
    for data in datas:
        use_lsp = 0
        for step in data["trajectory_steps"]:
            action = step["action"]
            function_blocks = function_block_regex.finditer(action)

            for block_match in function_blocks:
                function_name = block_match.group(1).strip()
                parameter_content = block_match.group(2)

            if "lsp_tool" in function_name:
                if "[status_code]:" in step["observation"]:
                    parameters_found = []
                    parameter_matches = parameter_regex.finditer(parameter_content)
                    
                    for param_match in parameter_matches:
                        param_name = param_match.group(1).strip()
                        param_value = param_match.group(2).strip()
                        parameters_found.append({
                            "name": param_name,
                            "value": param_value
                        })
                    
                    if parameters_found[0]["value"] == "get_workspace_symbols":
                        if ("def " in parameters_found[1]["value"] or "class " in parameters_found[1]["value"]) and "No data returned." in step["observation"]:
                            data["trajectory_steps"].remove(step)
                        else:
                            use_lsp += 1
                    else:
                        use_lsp += 1
                else:
                    use_lsp = 0
                    data["trajectory_steps"].remove(step)

        if use_lsp >= limit_lsp and data["reward"] == 1.0:
            filtered_datas.append(data)

    return filtered_datas


def process_llm_as_judge_datas(datas):
    trajectories = {}
    for data in datas:
        if 'instance_id' not in data['ds']:
            instance_id = data['ds']['docker_image'].split(":")[-1].split(".txt")[0]
            data['ds']['instance_id'] = instance_id
        else:
            instance_id = data['ds']['instance_id']
        messages = []
        agent_args = data["agent_args"]
        trajectory_steps = data['trajectory_steps']
        
        sp_1 = agent_args["system_prompt"]
        sp_2 = agent_args["instance_prompt"]
        messages.append({"role": "system", "content": sp_1})
        messages.append({"role": "user", "content": sp_2})
        
        for step in trajectory_steps:
            thought = step["thought"].replace("\n", " ")
            action = step["action"].replace("\n", " ")
            step_idx = step['step_idx']
            observation = step["observation"]
            
            if "lsp_tool" not in action:
                if len(observation) > 100:
                    observation = observation[:100] + "..."
            
            if thought:
                content = thought + "\n\n" + action
            else:
                content = action
            
            observation = observation.strip("\n")
            messages.append({"step": step_idx, "role": "assistant", "content": content})
            messages.append({"step": step_idx, "role": "tool", "content": observation})
        
        trajectories[instance_id] = messages
    
    return trajectories


def model_query(messages, logger, instance_id, model_name, api_base, api_key):
    """Query the model with retries"""
    retries = 0
    max_retries = 3
    
    while retries < max_retries:
        try:
            logger.info(f"[{instance_id}] Starting LLM query (Attempt {retries + 1}/{max_retries})")
            response = litellm.completion(
                model=model_name,
                messages=messages,
                timeout=180,
                api_base=api_base,
                api_key=api_key,
                max_tokens=16384,
            )

            logger.info(f"[{instance_id}] LLM query successful")
            return response
            
        except Exception as e:
            logger.error(f"[{instance_id}] LLM query failed (Attempt {retries + 1}/{max_retries}): {e}")
            retries += 1
            time.sleep(5)
            if "RateLimitError" in str(e):
                logger.warning(f"[{instance_id}] Rate limit encountered, waiting 10 seconds")
                time.sleep(10)
            if retries >= max_retries:
                logger.error(f"[{instance_id}] Maximum retries reached, giving up")
                raise e


def extract_answer_from_response(response_text):
    """Extract yes/no answer from LLM response"""
    boxed_pattern = r'\\boxed\{(yes|no)\}'
    match = re.search(boxed_pattern, response_text, re.IGNORECASE)
    
    if match:
        return match.group(1).lower()
    
    if "answer: yes" in response_text.lower() or "answer:** yes" in response_text.lower():
        return "yes"
    elif "answer: no" in response_text.lower() or "answer:** no" in response_text.lower():
        return "no"
    
    return "unknown"


def llm_as_judge(filtered_datas, logger, spec_id, max_workers, model_name, api_base, api_key):
    """Evaluate LSP tool usage effectiveness using LLMs (Parallel Version)"""
    logger.info(f"Starting LLM evaluation for {len(filtered_datas)} instances")
    trajectories = process_llm_as_judge_datas(filtered_datas)
    
    final_trajectories = []
    results_lock = Lock()
    
    def process_single_trajectory(instance_id, trajectory_messages, spec_id):
        """Function to process a single trajectory"""
        logger.info(f"[{instance_id}] Starting process")
        
        trajectory_str = ""
        for msg in trajectory_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            step = msg.get("step", "")
            
            if step:
                trajectory_str += f"\n[Step {step}] [{role.upper()}]: {content}\n"
            else:
                trajectory_str += f"[{role.upper()}]: {content}\n"
        
        judge_prompt = LLM_JUDGE_PROMPT.format(trajectory=trajectory_str)
        judge_messages = [{"role": "user", "content": judge_prompt}]
        
        try:
            response = model_query(judge_messages, logger, instance_id, model_name, api_base, api_key)
            analysis_text = response.choices[0].message.content
            answer = extract_answer_from_response(analysis_text)
            
            logger.info(f"[{instance_id}] Evaluation completed, Result: {answer}")
            logger.info(f"[{instance_id}] Analysis Summary:\n{analysis_text[:500]}...")
            
            directory = f"llm_as_judge_logs/{spec_id}/"
            if directory:
                os.makedirs(directory, exist_ok=True)
            result_file = f"{directory}/analysis_{instance_id}.txt"
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"Instance ID: {instance_id}\n")
                f.write(f"Answer: {answer}\n")
                f.write("=" * 80 + "\n")
                f.write(analysis_text)
            logger.info(f"[{instance_id}] Full analysis saved to: {result_file}")
            
            result = {
                "instance_id": instance_id,
                "analysis": analysis_text,
                "answer": answer,
                "original_data": next(d for d in filtered_datas if d['ds']['instance_id'] == instance_id)
            }
            
            tqdm.write(f"✓ {instance_id}: {answer}")
            return result
            
        except Exception as e:
            logger.error(f"[{instance_id}] Evaluation failed: {str(e)}", exc_info=True)
            
            result = {
                "instance_id": instance_id,
                "analysis": f"Error during analysis: {str(e)}",
                "answer": "error",
                "original_data": next(d for d in filtered_datas if d['ds']['instance_id'] == instance_id)
            }
            
            tqdm.write(f"✗ {instance_id}: Evaluation failed - {str(e)}")
            return result
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance = {
            executor.submit(process_single_trajectory, instance_id, trajectory_messages, spec_id): instance_id
            for instance_id, trajectory_messages in trajectories.items()
        }
        
        with tqdm(total=len(trajectories), desc="Evaluation Progress", unit="instance") as pbar:
            for future in as_completed(future_to_instance):
                result = future.result()
                
                with results_lock:
                    final_trajectories.append(result)
                
                pbar.update(1)
    
    logger.info(f"LLM evaluation completed, processed {len(final_trajectories)} instances in total")
    return final_trajectories


def save_to_path(output_path, filtered_datas, final_trajectories, logger):
    saved_trajs = []
    
    for result in final_trajectories:
        if result["answer"] == 'yes':
            for traj in filtered_datas:
                if traj['ds']['instance_id'] == result['instance_id']:
                    saved_trajs.append(traj)
                    logger.info(f"Saving valid data: {result['instance_id']}")
                    break

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in saved_trajs:
            json_str = json.dumps(entry, ensure_ascii=False)
            f.write(json_str + '\n')
    
    logger.info(f"Total {len(saved_trajs)} valid data entries saved to: {output_path}")


def save_to_path_original(output_path, filtered_datas, logger):

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in filtered_datas:
            json_str = json.dumps(entry, ensure_ascii=False)
            f.write(json_str + '\n')
    
    logger.info(f"Total {len(filtered_datas)} valid data entries before llm_as_judge saved to: {output_path}")



def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the effectiveness of LSP tools in solving code problems using LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input JSONL file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output JSONL file path (optional, default: auto-generated)'
    )
    
    parser.add_argument(
        '--limit-lsp',
        type=int,
        default=2,
        help='Minimum LSP tool usage threshold (default: 2)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum number of threads for parallel processing (default: 5)'
    )
    
    parser.add_argument(
        '--model',
        default='openai/nbgexp',
        help='LLM model name (default: openai/nbgexp)'
    )
    
    parser.add_argument(
        '--api-base',
        default='xx',
        help='API base URL'
    )
    
    parser.add_argument(
        '--api-key',
        default='xx',
        help='API key'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger, log_file = setup_logging()
    
    file_path = args.input
    limit_lsp = args.limit_lsp
    
    if not os.path.exists(file_path):
        logger.error(f"Input file does not exist: {file_path}")
        return
    
    spec_id = os.path.basename(file_path).replace('.jsonl', '')
    spec_dir = os.path.basename(os.path.dirname(file_path))
    
    if args.output:
        output_path = args.output
        output_dir = os.path.dirname(output_path)
    else:
        # base_dir = os.path.dirname(os.path.dirname(file_path))
        base_dir = "./R2E-Gym/sft_data"
        output_dir = os.path.join(base_dir, spec_dir, f'llm_judge_filter_beyond_{limit_lsp}')
        output_path = os.path.join(output_dir, f'{spec_id}.jsonl')
        output_original_dir = os.path.join(base_dir, spec_dir, f'rule_based_filter_beyond_{limit_lsp}')
        output_original_path = os.path.join(output_original_dir, f'{spec_id}.jsonl')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_original_dir, exist_ok=True)

    logger.info("="*80)
    logger.info("Starting LSP Tool Evaluation Process")
    logger.info(f"Input File: {file_path}")
    logger.info(f"Output File: {output_path}")
    logger.info(f"LSP Usage Threshold: {limit_lsp}")
    logger.info(f"Parallel Workers: {args.max_workers}")
    logger.info(f"Using Model: {args.model}")
    logger.info("="*80)

    logger.info("Reading data...")
    datas = read_datas(file_path)
    logger.info(f"Total data count: {len(datas)}")
    
    logger.info("Filtering data...")
    filtered_datas = rule_filter_datas(datas, logger, limit_lsp)
    logger.info(f"Filtered data count: {len(filtered_datas)}")
    
    if not filtered_datas:
        logger.warning("No data meeting the criteria, process terminated")
        return

    save_to_path_original(output_original_path, filtered_datas, logger)


    logger.info("Starting LLM evaluation...")
    final_trajectories = llm_as_judge(
        filtered_datas, 
        logger, 
        spec_id,
        args.max_workers,
        args.model,
        args.api_base,
        args.api_key
    )
    
    logger.info(f"Saving results to: {output_path}")
    save_to_path(output_path, filtered_datas, final_trajectories, logger)

    yes_count = sum(1 for t in final_trajectories if t['answer'] == 'yes')
    no_count = sum(1 for t in final_trajectories if t['answer'] == 'no')
    unknown_count = sum(1 for t in final_trajectories if t['answer'] == 'unknown')
    error_count = sum(1 for t in final_trajectories if t['answer'] == 'error')
    
    logger.info("="*80)
    logger.info("Evaluation Statistics:")
    logger.info(f"  Yes (LSP helpful): {yes_count}")
    logger.info(f"  No (LSP unhelpful): {no_count}")
    logger.info(f"  Unknown (Undetermined): {unknown_count}")
    logger.info(f"  Error (Evaluation error): {error_count}")
    logger.info(f"  Total: {len(final_trajectories)}")
    logger.info("="*80)
    logger.info(f"Full log saved to: {log_file}")
    logger.info("Process completed!")


if __name__ == "__main__":
    main()