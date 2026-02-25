import json
import os
import pandas as pd
import re
import html

FC_SP = '''## Function Definition
- You have access to the following functions:

---- BEGIN FUNCTION #1: execute_bash ----
**Description**: Execute a bash command in the terminal within a persistent shell session.
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
* Soft timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command
* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts or commands in this environment. The runtime may not support them and can cause unusable shell sessions. If you want to run multi-line bash commands, write the commands to a file and then run it, instead.
* For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.
* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.

**Parameters**:
(1) command (string, required): The bash command to execute. For example: `python my_script.py`. If not provided, will show help. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together."
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: str_replace_editor ----
**Description**: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* The following binary file extensions can be viewed in Markdown format: [\".xlsx\", \".pptx\", \".wav\", \".mp3\", \".m4a\", \".flac\", \".pdf\", \".docx\"]. IT DOES NOT HANDLE IMAGES.\n* The create command cannot be used if the specified path already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique.
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
* This tool can be used for creating and editing files in plain-text format.
* Before using this tool:
- 1. Use the view tool to understand the file's contents and context
- 2. Verify the directory path is correct (only applicable when creating new files):
    - Use the view tool to verify the parent directory exists and is the correct location
    - When making edits:
    - Ensure the edit results in idiomatic, correct code
    - Do not leave the code in a broken state
    - Always use absolute file paths (starting with /)
* Remember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.

**Parameters**:
(1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`.
(2) path (string, required): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
(3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
(4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
(5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
(6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
(7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
---- END FUNCTION #2 ----


---- BEGIN FUNCTION #3: submit ----
**Description**: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
No parameters are required for this function.
---- END FUNCTION #3 ----

- If you choose to call a function, ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task.
</IMPORTANT>
'''
bc_en_question =set()



def process_single_jsonl(jsonl_f):
    messages_all = []
    tot_count = 0
    with open(jsonl_f,"r") as f:
        for line in f:
            tot_count+=1
            messages = []
            data=json.loads(line)
            reward = data["reward"]
            # print(data.keys())
            # print(data["trajectory_steps"][0].keys())
            # exit()
            # if reward !=1:
            #     continue
            problem_statement = data["problem_statement"]
            agent_args = data["agent_args"]
            trajectory_steps = data["trajectory_steps"]
            # print(data["step_count"])
            step_count =trajectory_steps[-1]["step_count"]
            token_usage_total =trajectory_steps[-1]["token_usage_total"]
            sp_1 = agent_args["system_prompt"]
            sp_2 = agent_args["instance_prompt"]
            
            sp_2 = sp_2.replace("{problem_statement}",problem_statement).replace("{working_dir}","/testbed")
            
            
            messages.append({"role":"system","content":sp_1})
            messages.append({"role":"system","content":FC_SP})
            messages.append({"role":"user","content":sp_2})
            for step in trajectory_steps:
                # for k,v in step.items():
                #     print(f"{k}: {[v]}")
                # print("=="*80)
                thought = step["thought"]
                # print(thought)
                # exit()
                # content_short = thought.strip().split("</think>")

                # if len(content_short)>=2:
                #     print(content_short)
                action = step["action"]
                observation = step["observation"]
                messages.append({"role":"assistant","content":thought+"\n\n"+action})
                messages.append({"role":"tool","content":observation})
            idx = jsonl_f +"@"+str(tot_count)
            messages_all.append([messages,step_count,token_usage_total,idx,reward])

        file_name = jsonl_f.split("/")[-1]
        if tot_count!=0:
            print(f"{file_name}:\nTotal Count: {tot_count}, Reward=1 Count: {len(messages_all)}, Retention Rate: {len(messages_all)/tot_count}")
            print("=="*50)
        else:
            print(f"{file_name}:\nTotal Count: {tot_count}")

    return messages_all

def process_folders_to_jsonl(folder_list, output_jsonl):
    """Takes multiple folders, processes all jsonl files within them, and merges the output."""
    merged_messages = []
    all_num= len(folder_list)
    now_idx = 0
    for folder in folder_list:
        print(f"Starting process: {folder}")
        now_idx +=1
        print(f"Processing: {now_idx}/{all_num}")
        for name in os.listdir(folder):
            if name.endswith(".jsonl"):
                full_path = os.path.join(folder, name)
                
                if "verified" in full_path and "smith" not in full_path:
                    print(f"Skipping test set file: {full_path}")
                    print(f"=="*50)
                    continue
                print(f"Processing: {full_path}")
                msgs = process_single_jsonl(full_path)
                merged_messages.extend(msgs)

    save_messages_to_jsonl(merged_messages, output_jsonl)
    print(f"\nDone! Total trajectories = {len(merged_messages)}")
    print(f"Saved to: {output_jsonl}")


def save_messages_to_jsonl(messages_all, output_jsonl):
    """Write messages_all into a jsonl file."""
    if not output_jsonl:
        return 
    else:
        with open(output_jsonl, "w") as f:
            for msgs in messages_all:
                # obj = {"train":{"input": msgs[0],
                # "token_usage_total": msgs[2],
                # "step_count": msgs[1]}}
                obj = {"input": msgs[0],
                "token_usage_total": msgs[2],
                "step_count": msgs[1],
                # "file_idx":msgs[3]
                "reward": msgs[-1]
                }
                
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            

def find_subfolders_with_str(folder_path: str, keyword: str):
    result = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if os.path.isdir(full_path) and keyword in name:
            result.append(full_path)
    return result

if __name__ == "__main__":
    
    

    folder_list = add_list = ["./R2E-Gym/results/1214_0.0001_0.4", "./R2E-Gym/results/1214_0.4_0.6"]
    

    print("The following data is banned, please verify carefully:") 
    for f in rm_list:
        print(f)
    print("=="*50)
    
    print("The following data is ready to be processed, please verify carefully:")   
    for f in folder_list:
        print(f)
    
    print("=="*50)
    print(f'Total number of files to be processed: {len(folder_list)}')

    output_jsonl ="./R2E-Gym/results/swe_all_w_sp_obs.jsonl"
    process_folders_to_jsonl(folder_list, output_jsonl)
