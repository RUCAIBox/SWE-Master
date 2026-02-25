from typing import Callable

import torch
from torch.utils.data import Dataset
ALLOWED_STR_REPLACE_EDITOR_COMMANDS = ["view", "create", "str_replace", "insert"]

_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

str_replace_editor_tool = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": _STR_REPLACE_EDITOR_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "description": f"The command to run. Allowed options are: {', '.join(f'`{cmd}`' for cmd in ALLOWED_STR_REPLACE_EDITOR_COMMANDS)}.",
                    "enum": ALLOWED_STR_REPLACE_EDITOR_COMMANDS,
                    "type": "string",
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                    "type": "string",
                },
                "file_text": {
                    "description": "Required for the `create` command, contains the content of the file to be created.",
                    "type": "string",
                },
                "old_str": {
                    "description": "Required for the `str_replace` command, specifies the string in `path` to replace.",
                    "type": "string",
                },
                "new_str": {
                    "description": "Optional for the `str_replace` command to specify the replacement string. Required for the `insert` command to specify the string to insert.",
                    "type": "string",
                },
                "insert_line": {
                    "description": "Required for the `insert` command. The `new_str` will be inserted AFTER the line specified.",
                    "type": "integer",
                },
                "view_range": {
                    "description": "Optional for the `view` command when `path` points to a file. Specifies the line range to view. E.g., [11, 12] shows lines 11 and 12. Indexing starts at 1. Use [start_line, -1] to show all lines from `start_line` to the end.",
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["command", "path"],
        },
    },
}


_R2EGYM_BASH_EXECUTE_DESCRIPTION = """
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string, required): The bash command to execute. For example: `python my_script.py`
"""

r2egym_bash_execute_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": _R2EGYM_BASH_EXECUTE_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The command (and optional arguments) to execute. For example: 'python my_script.py'",
                }
            },
            "required": ["cmd"],
        },
    },
}

_SUBMIT_DESCRIPTION = """
A simple submit tool to finish tasks.

This tool signals completion of a task or submission of results.
No parameters required - simply call to indicate task completion.
"""

submit_tool = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": _SUBMIT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}



_BASH_DESCRIPTION = """
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string, optional): The bash command to execute. For example: `python my_script.py`. If not provided, will show help.
"""

execute_bash_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": _BASH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command (and optional arguments) to execute. For example: 'python my_script.py'",
                }
            },
            "required": [],
        },
    },
}

tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]


# from openrlhf.utils.utils import zero_pad_sequences

# keep support for conversations style
def preprocess_data(
    data, input_template=None, input_key="input", output_key=None, apply_chat_template=None, multiturn=False
):
    if apply_chat_template:
        if output_key:
            kill

        else:
            prompt = apply_chat_template(data[input_key][:-1], tools = tools, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tools = tools, tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=12,  # Specify the number of processors you want to use
        multiturn=False,
        tokenizer_path = "",
        data_file = ""
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiturn = multiturn

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", True)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
        print(dataset)
        # exit()
        # Parallel loading datasets

        print(f"tokenizer path: {tokenizer_path}")
        tokenzier_name = tokenizer_path.split("/")[-1]
        file_name = data_file.split("/")[-1].strip("jsonl")
        out_file = f"./OpenRLHF_SFT/SFT_data_pre_process/{file_name}_processed_{tokenzier_name}_w_fc.jsonl"
        print(f"out_file path: {out_file}")

        processed_dataset = dataset.map(
            self.process_data,
            # remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        print(processed_dataset)
        
        with open(out_file, "w", encoding="utf-8") as f:
            for sample in processed_dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"Finish Writing {os.path.abspath(out_file)}, {len(processed_dataset)} lines")
        exit()
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
        
        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None

    def process_data(self, data):
        if self.multiturn and self.output_key:
            data[self.input_key].append(data[self.output_key])
            data[self.output_key] = None

        if self.multiturn:
            assert (
                not self.output_key or not data[self.output_key]
            ), "You should put the whole trajactory into data[input_key] and do not set output_key"
            input_key = self.input_key
            apply_chat_template = self.apply_chat_template

            response_ranges = []
            for idx, message in enumerate(data[input_key]):
                if message["role"] == "assistant":
                    prompt = apply_chat_template(data[input_key][:idx], tools = tools, tokenize=False, add_generation_prompt=True)
                    response = apply_chat_template(data[input_key][: idx + 1], tools = tools, tokenize=False)[len(prompt) :]

                    start_idx = (
                        self.tokenizer(
                            prompt,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                    )

                    end_idx = (
                        start_idx
                        + self.tokenizer(
                            response,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                        - 1
                    )
                    response_ranges.append((start_idx, end_idx))  # left close right close
        
        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
        ) #Response is the final turn here

        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = 0

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
            "response_ranges": response_ranges if self.multiturn else None,
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        if not self.pretrain_mode:
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
            # print(text)
            # exit()
        else:
            text = prompt

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = input_token["input_ids"]
        attention_mask = input_token["attention_mask"]
        loss_mask = self.get_loss_mask(input_ids, idx)

        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_ids[0][-1] = self.tokenizer.eos_token_id
            attention_mask[0][-1] = True
        return input_ids, attention_mask, loss_mask

    def get_loss_mask(self, input_ids, idx):
        if self.pretrain_mode:
            return torch.ones_like(input_ids, dtype=torch.float32)  # shape:[1, seq_len]

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if not self.multiturn:
            prompt_ids_len = self.prompt_ids_lens[idx]
            loss_mask[0, prompt_ids_len - 1 : -1] = 1
        else:
            response_ranges = self.response_ranges[idx]
            for start_idx, end_idx in response_ranges:
                loss_mask[0, start_idx - 1 : end_idx] = 1
        return loss_mask

    # def collate_fn(self, item_list):
    #     input_ids = []
    #     attention_masks = []
    #     loss_masks = []

    #     for input_id, attention_mask, loss_mask in item_list:
    #         input_ids.append(input_id)
    #         attention_masks.append(attention_mask)
    #         loss_masks.append(loss_mask)

    #     input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
    #     attention_masks = zero_pad_sequences(attention_masks, "right")
    #     loss_masks = zero_pad_sequences(loss_masks, "right")
    #     return input_ids, attention_masks, loss_masks

if __name__ == "__main__":
    import json
    import os
    from datasets import load_dataset
    def blending_datasets(
        dataset,
        probabilities=None,
        strategy=None,
        seed=42,
        max_count=1e8,
        stopping_strategy="all_exhausted",
        dataset_split="train",
    ):

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        
        if ext in [".json", ".jsonl", ".csv", ".parquet", ".arrow"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            if dataset_split and dataset_split in data:
                data = data[dataset_split]
            dataset = data
            # strategy.print(f"loaded {dataset} with data_files={dataset}")            
        else:
            kill

        return dataset

    def postprocess_r2egym_traj(jsonl_f):
        messages_all = []
        with open(jsonl_f,"r") as f:
            for line in f:
                messages = []
                data=json.loads(line)

                agent_args = data["agent_args"]
                trajectory_steps = data["trajectory_steps"]

                sp_1 = agent_args["system_prompt"]
                sp_2 = agent_args["instance_prompt"]
                messages.append({"role":"system","content":sp_1})
                messages.append({"role":"user","content":sp_2})
                for step in trajectory_steps:
                    thought = step["thought"]
                    action = step["action"]
                    observation = step["observation"]
                    messages.append({"role":"assistant","content":thought+"\n\n"+action})
                    messages.append({"role":"tool","content":observation})

                messages_all.append(messages)

        return messages_all
    def postprocess_sft_data(jsonl_f):
        messages_all = []
        with open(jsonl_f,"r") as f:
            for line in f:
                data=json.loads(line)
                messages_all.append(data["input"])

        return messages_all

    print("====Begin!!====")
    from transformers import AutoTokenizer
    from datasets import Dataset


    data_file = "./OpenRLHF_SFT/scripts_swe_master/sft_data_demo.jsonl"
    train_data = blending_datasets(data_file)
    print(train_data)

    tokenizer_path ="./models/Qwen2.5-Coder-32B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    class Args:
        input_key = "input"
        output_key = None
    class Strategy:
        args = Args()
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        131072, #args.max_len,
        strategy=Strategy(),
        multiturn=True,  # test multi turn
        pretrain_mode=False,
        tokenizer_path = tokenizer_path,
        data_file =data_file
    )
       