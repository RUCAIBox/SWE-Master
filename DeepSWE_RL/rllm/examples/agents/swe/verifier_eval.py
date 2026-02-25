import concurrent.futures
import re
from collections import defaultdict

import datasets
import litellm
import numpy as np
import pandas as pd
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm


#########################################################
# condenser
#########################################################
def _count_tokens(messages: list[dict[str, str]], llm_name="gpt4o") -> int:
    """
    Counts the tokens for a list of messages using the litellm library.
    Adjust as needed depending on the model and library.
    """
    token_count = litellm.token_counter(model=llm_name, messages=messages)
    # print(f"Total tokens in conversation: {token_count}")
    return token_count


def dummy_count_tokens(s: str) -> int:
    """
    Dummy token counter that counts words.
    Replace this with your actual token counting logic.
    """
    messages = [{"role": "system", "content": "Identify whether the following agent trajectory is correct or not. Answer 'YES' or 'NO'"}, {"role": "user", "content": s}]

    return _count_tokens(messages)


def condense(input_str: str, max_tokens: int = 25000) -> str:
    """
    If the token count of input_str exceeds max_tokens, then starting with the second
    [USER]...[/USER] block (the oldest after the first), replace its inner content with
    a placeholder until the total token count is under the limit.

    The first [USER] block is left intact.
    """
    placeholder = "<Observation condensed for saving context>"

    # Check initial token count
    if dummy_count_tokens(input_str) <= max_tokens:
        return input_str

    # Regex to match [USER] blocks
    pattern = re.compile(r"(\[USER\])(.*?)(\[/USER\])", re.DOTALL)

    new_str = input_str
    # Continue condensing until token count is below the threshold or nothing changes.
    while dummy_count_tokens(new_str) > max_tokens:
        # Re-find all [USER] blocks in the updated string
        matches = list(pattern.finditer(new_str))
        if len(matches) <= 1:
            # Nothing more to condense (either no [USER] blocks or only one exists)
            break

        replaced = False
        # Iterate over all [USER] blocks starting from the second one
        for i, m in enumerate(matches):
            if i == 0:
                continue  # leave the first [USER] block unchanged
            # If already condensed, skip it
            if m.group(2).strip() == placeholder:
                continue
            # Build the new block with condensed content
            new_block = m.group(1) + placeholder + m.group(3)
            # Replace this block in the string using its current indices
            start, end = m.start(), m.end()
            new_str = new_str[:start] + new_block + new_str[end:]
            replaced = True
            # Break out after replacing one block so we can re-check token count
            break
        if not replaced:
            # All subsequent [USER] blocks are already condensed
            break

    return new_str


# sudo docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v /home/ubuntu/LLaMA-Factory/saves:/models -p 8000:8000     --ipc=host     vllm/vllm-openai:latest --model Qwen/Qwen2.5-Coder-32B-instruct --lora-modules model1=/models/qwen25coder-32b-instruct-reasoning_patch_verif_2kcontext_lora40_3epoch_lr1en5-v1/ model2=/models/qwen25coder-32b-instruct-reasoning_patch_verif_2kcontext_lora40_3epoch_lr1en4-v1 model3=/models/qwen25coder-32b-instruct-reasoning_patch_verif_2kcontext_lora40_4epoch_lr1en5-v1 model4=/models/qwen25coder-32b-instruct-reasoning_patch_verif_2kcontext_lora40_lr1en5-v1 --enable-lora --host 0.0.0.0 --port 8000 --tensor-parallel-size 8 --gpu-memory-utilization 0.95 --dtype bfloat16 --enable-prefix-caching  --max-lora-rank 64 --max-logprobs 500


#########################################################
# main eval verifier
#########################################################
class ReasoningVerifierArgs(BaseModel):
    # llm_name: str = 'hosted_vllm//data/home/jaskirats/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-verifier-sonnet_32b_gpt4o_combined_32k_verifier_ep2-32k-v1'
    llm_name: str = "hosted_vllm//data/home/jaskirats/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-verifier-sonnet_32b_gpt4o_combined_32k_verifiernew_reasoning_ep2-32k-v1"
    # llm_name: str = 'hosted_vllm//data/home/jaskirats/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-32b-instruct-verifier-sonnet_32b_gpt4o_combined_32k_verifiernew_reasoning_ep2-20k-v1'
    # llm_name: str = 'hosted_vllm//data/home/jaskirats/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-verifier-sonnet_32b_gpt4o_combined_32k_verifier_ep2-32k-v1'
    temperature: float = 0.0
    num_samples: int = 1
    max_retries: int = 3
    # eval_dataset: str = "r2e-edits/32b_swebv_temp08_10_verifier"
    eval_dataset: str = "r2e-edits/32b_swebv_temp08_10_patch_verifier"
    out_file: str = "results_traj_verifiernew_p2p-14B-v1.csv"
    port: int = 8000


def run_model(arg) -> list[str]:
    message_list: list[dict]
    args: ReasoningVerifierArgs

    message_list, args = arg

    llm_name = args.llm_name
    temperature = args.temperature
    num_samples = args.num_samples

    retries = 0

    # condense messages
    condensed_user_msg = message_list[1]["content"]  # condense(input_str=message_list[1]['content'], max_tokens = 28000)
    message_list = [{"role": "system", "content": message_list[0]["content"]}, {"role": "user", "content": condensed_user_msg}]
    # query the model with retries
    while retries < args.max_retries:
        try:
            response = litellm.completion(
                model=llm_name,
                tools=[],
                messages=message_list,
                n=num_samples,
                function_call=None,
                tool_choice="none",
                timeout=120,
                api_key=None,
                temperature=temperature,
                # api_base="http://localhost:8000/v1",
                api_base=f"http://localhost:{args.port}/v1",
                vertex_ai_project="r2eg-441800",
                vertex_ai_location="europe-west1",
                logprobs=True,
                # extra_body={
                #     "guided_choice": ["YES", "NO"]
                # },
                top_logprobs=20,
            )
            break
        except Exception as e:
            print(f"LLM query failed: {e}")
            retries += 1
            if retries >= args.max_retries:
                raise e
    """
    output is of the form:
    [REASON] {reasoning} [/REASON] [ANSWER] {answer} [/ANSWER]
    we extract the answer and then collect the corresponding logits for YES/NO
    """
    # outputs = [response.choices[i].message.content for i in range(num_samples)]

    yes_probs = []

    for i in range(num_samples):
        # tokens = [x["token"] for x in response.choices[i].logprobs["content"]]
        # logits = [x["logprob"] for x in response.choices[i].logprobs["content"]]
        # print (response.choices[0].logprobs.content[0].top_logprobs)
        all_logits = [{lp.token: lp.logprob for lp in response.choices[0].logprobs.content[4].top_logprobs}]
        # all_logits = [
        #     {lp["token"]: lp["logprob"] for lp in x}
        #     for x in response.choices[i].logprobs["content"][0]["top_logprobs"]
        # ]

        # print(outputs[i])
        # print(tokens)

        # find YES / NO in the string after the tokens
        # j = 4
        # for j in range(len(tokens) - 1):
        #     if tokens[j] == 'YES' or tokens[j] == 'NO':
        #         break
        # if tokens[j] == ">":
        #     if tokens[j + 1] == "ANS" and tokens[j + 2] == "WER":
        #         break

        # try:
        #     k = tokens.index("YES", j)
        # except ValueError:
        #     try:
        #         k = tokens.index("NO", j)
        #     except ValueError:
        #         # try:
        #         #     k = tokens.index("YES", j)
        #         # except ValueError:
        #         #     try:
        #         #         k = tokens.index("NO", j)
        #         #     except ValueError:
        #         #         k = -1
        #         #         yes_probs.append(-1)
        #         #         continue
        #         k = -1
        #         yes_probs.append(-1)
        #         continue

        k = 0
        # print(all_logits[k])
        p_yes = all_logits[k].get("YES", -10000)
        p_no = all_logits[k].get("NO", -10000)
        # print(p_yes, p_no)
        yes_probs.append((np.exp(p_yes)) / (np.exp(p_yes) + np.exp(p_no)))
        # print(yes_probs[-1])

    return yes_probs


def eval_row(row: dict, args: ReasoningVerifierArgs):
    message = list(row["messages"][:2])
    try:
        yes_probs = run_model((message, args))
        yes_probs = [x for x in yes_probs if x != -1]
        if len(yes_probs) == 0:
            return (row["docker_images"], None, None)
        avg_yes_prob = float(sum(yes_probs) / len(yes_probs))
    except Exception:
        avg_yes_prob = 0
    gt_correct = row["rewards"] == 1
    return (row["docker_images"], avg_yes_prob, gt_correct)


def flush_to_csv(results_data, file_path, header_cols, write_header=False):
    """
    Writes accumulated results_data to CSV in 'append' mode.
    If write_header=True, it writes the column header (first flush).
    """
    if not results_data:
        return  # Nothing to flush
    df_temp = pd.DataFrame(results_data, columns=header_cols)
    df_temp.to_csv(file_path, mode="a", header=write_header, index=False)
    # Clear the list after writing
    results_data.clear()


def main(args):
    # 1. Load dataset into a DataFrame
    dataset = datasets.load_dataset(args.eval_dataset)
    df = dataset["train"].to_pandas()

    # 2. Define which columns to keep (everything except "messages")
    original_cols = [col for col in df.columns if col != "messages"]
    # We'll add these new columns for the final output
    new_cols = ["avg_yes_prob", "gt_correct"]
    header_cols = original_cols + new_cols

    # 3. Prepare to store results
    results_data = []  # in-memory list of dicts/rows
    BATCH_SIZE = 128  # how often to flush to CSV

    output_file = args.out_file  # e.g. "results.csv"

    # 4. Before processing, open CSV in write mode to create an empty file
    #    and write the header on the first flush.
    open(output_file, "w").close()  # ensure file is empty

    # 5. Evaluate each row in parallel
    rows = [row for _, row in df.iterrows()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures_map = {executor.submit(eval_row, row, args): i for i, row in enumerate(rows)}

        # We’ll track how many we’ve processed so we know when to flush
        processed_count = 0
        for future in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map)):
            i = futures_map[future]
            row = rows[i]
            docker_images, avg_yes_prob, gt_correct = future.result()

            # Build a dict for this row, excluding "messages"
            row_dict = {}
            for col in original_cols:
                row_dict[col] = row[col]

            # Add newly computed fields
            row_dict["avg_yes_prob"] = avg_yes_prob
            row_dict["gt_correct"] = gt_correct

            results_data.append(row_dict)
            processed_count += 1

            # Flush to CSV if we hit the batch size
            if processed_count % BATCH_SIZE == 0:
                # Write to CSV in append mode
                # If this is the first flush (processed_count == BATCH_SIZE), write header
                write_header = processed_count == BATCH_SIZE
                flush_to_csv(results_data, output_file, header_cols, write_header=write_header)

        # After the loop, flush any leftover rows that didn't reach the batch size
        if results_data:
            # If no flush has happened yet, we do want the header
            write_header = processed_count <= BATCH_SIZE
            flush_to_csv(results_data, output_file, header_cols, write_header=write_header)

    # 6. Optionally, do post-processing/aggregation on *all results* in memory.
    #    If you only want to do final aggregation, you can do so on a DataFrame
    #    created from the same `rows` you have in memory (or re-read from CSV).
    #    Below, we'll just convert everything we processed into a final DataFrame
    #    in memory. (We already cleared `results_data` on flush, so let's re-read
    #    from CSV or track them in another list if you want the entire data.)

    # Example: re-read from CSV for final aggregator
    final_df = pd.read_csv(output_file)

    # Aggregation example:
    # (a) Max avg_yes_prob per docker_images
    aggregated_max_yes = final_df.groupby("docker_images", as_index=False).apply(lambda g: g.loc[g["avg_yes_prob"].idxmax()]).reset_index(drop=True)
    overall_accuracy_max_yes = aggregated_max_yes["gt_correct"].mean()

    # (b) Min num_steps
    aggregated_min_steps = final_df.groupby("docker_images", as_index=False).apply(lambda g: g.loc[g["num_steps"].idxmin()]).reset_index(drop=True)
    overall_accuracy_min_steps = aggregated_min_steps["gt_correct"].mean()

    # (c) Min patch_size
    aggregated_min_patch = final_df.groupby("docker_images", as_index=False).apply(lambda g: g.loc[g["patch_size"].idxmin()]).reset_index(drop=True)
    overall_accuracy_min_patch = aggregated_min_patch["gt_correct"].mean()

    # (d) Max p2p_rate
    aggregated_max_p2p = final_df.groupby("docker_images", as_index=False).apply(lambda g: g.loc[g["p2p_rate"].idxmax()]).reset_index(drop=True)
    overall_accuracy_max_p2p = aggregated_max_p2p["gt_correct"].mean()

    print("Total number of rows in final CSV:", len(final_df))
    print("Accuracy (max avg_yes_prob):", overall_accuracy_max_yes)
    print("Accuracy (min num_steps):", overall_accuracy_min_steps)
    print("Accuracy (min patch_size):", overall_accuracy_min_patch)
    print("Accuracy (max p2p_rate):", overall_accuracy_max_p2p)


def main_old(args: ReasoningVerifierArgs):
    dataset = datasets.load_dataset(args.eval_dataset)
    dataset = dataset["train"].to_pandas()  # .iloc[:200]
    rows = [row[1] for row in dataset.iterrows()]
    print(eval_row(rows[0], args))

    # Write header to file
    output_file = args.out_file  # "results.csv"
    with open(output_file, "w") as f:
        f.write("docker_images,avg_yes_prob,gt_correct\n")

    dockerwise_results = defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures_map = {executor.submit(eval_row, row, args): row for row in rows}
        for future in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map)):
            docker_images, avg_yes_prob, gt_correct = future.result()
            if avg_yes_prob is not None:
                dockerwise_results[docker_images].append((avg_yes_prob, gt_correct))

            # Save each result immediately
            with open(output_file, "a") as f:
                f.write(f"{docker_images},{avg_yes_prob},{gt_correct}\n")

    #  can still do aggregation if needed after saving individual results.
    all_results = []
    for docker_images, results in dockerwise_results.items():
        max_yes_prob = max([x[0] for x in results])
        max_yes_idx = [x[0] for x in results].index(max_yes_prob)
        max_yes_correct = results[max_yes_idx][1]
        correct = max_yes_correct
        all_results.append(correct)
        print(len(all_results), np.mean(all_results))


# def main(args: ReasoningVerifierArgs):
#     dataset = datasets.load_dataset(
#         args.eval_dataset,
#     )
#     dataset = dataset["train"].to_pandas().iloc[:200]

#     rows = dataset.iterrows()
#     rows = [row[1] for row in rows]

#     print(eval_row(rows[0], args))
#     # exit()

#     dockerwise_results = defaultdict(list)
#     with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
#         # futures = executor.map(lambda row: eval_row(row, args), rows)

#         # for docker_images, avg_yes_prob, gt_correct in tqdm(futures, total=len(rows)):
#         #     if avg_yes_prob is not None:
#         #         dockerwise_results[docker_images].append((avg_yes_prob, gt_correct))

#         futures_map = {executor.submit(eval_row, row, args): row for row in rows}

#         for future in tqdm(
#             concurrent.futures.as_completed(futures_map), total=len(futures_map)
#         ):
#             row = futures_map[future]
#             docker_images, avg_yes_prob, gt_correct = future.result()
#             if avg_yes_prob is not None:
#                 dockerwise_results[docker_images].append((avg_yes_prob, gt_correct))

#     all_results = []
#     for docker_images, results in dockerwise_results.items():
#         max_yes_prob = max([x[0] for x in results])
#         max_yes_idx = [x[0] for x in results].index(max_yes_prob)
#         max_yes_correct = results[max_yes_idx][1]

#         correct = max_yes_correct

#         all_results.append(correct)
#         print(len(all_results), np.mean(all_results))


if __name__ == "__main__":
    args = Fire(ReasoningVerifierArgs)
    main(args)
