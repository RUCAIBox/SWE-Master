from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.utils.utils import zero_pad_sequences

# keep support for conversations style
def preprocess_data(
    data, input_template=None, input_key="input", output_key=None, apply_chat_template=None, multiturn=False
):
    if apply_chat_template:
        if output_key:
            kill

        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        kill
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
        num_processors=16,  # Specify the number of processors you want to use
        multiturn=False,
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

        # Parallel loading datasets
        if not self.multiturn:
            processed_dataset = dataset.map(
                self.process_data,
                remove_columns=dataset.column_names,
                num_proc=num_processors,
            )
            
            # processed_dataset = dataset
            processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

            # Store the processed data in class attributes
            self.prompts = processed_dataset["prompt"]
            self.responses = processed_dataset["response"]
            self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
            self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None
            
        else: # multiturn
            processed_dataset = dataset
            processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
            self.prompts = processed_dataset["prompt"]
            self.responses = processed_dataset["response"]
            self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
            self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None



    def process_data(self, data):
        # kill
        #  data: dict or batch
        if not hasattr(self, "_process_count"):
            self._process_count = 0
            self._total = len(data["some_key"]) if "some_key" in data else 0  
        self._process_count += 1
        if self._process_count % 100 == 0:
            print(f"Processed {self._process_count} samples"*10)

        if self.multiturn and self.output_key:
            data[self.input_key].append(data[self.output_key])
            data[self.output_key] = None

        if self.multiturn:
            assert (
                not self.output_key or not data[self.output_key]
            ), "You should put the whole trajactory into data[input_key] and do not set output_key"
            input_key = self.input_key
            apply_chat_template = self.apply_chat_template
            # print(self.apply_chat_template)
            # response_ranges = []
            # response_prompts = []
            # response_texts = []
            # assistant_indices = []
            # for idx, message in enumerate(data[input_key]):
            #     if message["role"] == "assistant":
            #         prompt = apply_chat_template(
            #             data[input_key][:idx],
            #             tokenize=False,
            #             add_generation_prompt=True,
            #         )
            #         response = apply_chat_template(
            #             data[input_key][:idx + 1],
            #             tokenize=False
            #         )[len(prompt):]

            #         response_prompts.append(prompt)
            #         response_texts.append(response)
            #         assistant_indices.append(idx)
            # prompt_tokens = self.tokenizer(
            #                 response_prompts,
            #                 max_length=self.max_length,
            #                 padding=False,
            #                 truncation=True,
            #                 return_tensors=None, 
            #                 add_special_tokens=False,
            #             )
            # response_tokens = self.tokenizer(
            #                 response_texts,
            #                 max_length=self.max_length,
            #                 padding=False,
            #                 truncation=True,
            #                 return_tensors=None, 
            #                 add_special_tokens=False,
            #             )

            # prompt_lens = [sum(x) for x in prompt_tokens["attention_mask"]]
            # response_lens = [sum(x) for x in response_tokens["attention_mask"]]

            # response_ranges = []
            # for p_len, r_len in zip(prompt_lens, response_lens):
            #     start_idx = p_len
            #     end_idx = p_len + r_len - 1
            #     response_ranges.append((start_idx, end_idx))

            # for idx, message in enumerate(data[input_key]):
            #     if message["role"] == "assistant":
            #         prompt = apply_chat_template(data[input_key][:idx], tokenize=False, add_generation_prompt=True)
            #         response = apply_chat_template(data[input_key][: idx + 1], tokenize=False)[len(prompt) :]

            #         start_idx = (
            #             self.tokenizer(
            #                 prompt,
            #                 max_length=self.max_length,
            #                 padding=False,
            #                 truncation=True,
            #                 return_tensors="pt",
            #                 add_special_tokens=False,
            #             )["attention_mask"]
            #             .int()
            #             .sum()
            #             .item()
            #         )

            #         end_idx = (
            #             start_idx
            #             + self.tokenizer(
            #                 response,
            #                 max_length=self.max_length,
            #                 padding=False,
            #                 truncation=True,
            #                 return_tensors="pt",
            #                 add_special_tokens=False,
            #             )["attention_mask"]
            #             .int()
            #             .sum()
            #             .item()
            #             - 1
            #         )
            #         response_ranges.append((start_idx, end_idx))  # left close right close
        
        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
        )
        # print(prompt)
        # kill

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
            # filter the sample whose length is greater than max_length (2 for answer length)
            # print((prompt_ids_len))
            # print(self.max_length)
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

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        loss_masks = []

        for input_id, attention_mask, loss_mask in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")
        return input_ids, attention_masks, loss_masks

if __name__ == "__main__":
    pass
