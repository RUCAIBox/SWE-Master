import pandas as pd

N = 500


def accuracy_topkyesprob_then_p2p(df, k=3):
    df = df.copy()
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df["p2p_rate_len"] = df["p2p_rates"]

    grouped = df.groupby("docker_images")
    final_corrects = []

    for docker_image, group in grouped:
        # Sort by avg_yes_prob descending and take top k rows
        topk = group.sort_values(by="avg_yes_prob", ascending=False).head(k)

        # Among the top k, select the row with the highest p2p_rate_len
        chosen_row = topk.loc[topk["p2p_rate_len"].idxmax()]
        final_corrects.append(chosen_row["gt_correct"])

    # Overall accuracy
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / N


def accuracy_topkp2p_then_yesprob(df, k=3):
    df = df.copy()
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df["p2p_rate_len"] = df["p2p_rates"]

    grouped = df.groupby("docker_images")
    final_corrects = []

    for docker_image, group in grouped:
        # Sort by p2p_rate_len descending and take the top k rows
        # topk = group.sort_values(by="p2p_rate_len", ascending=False).head(k)
        topk = get_topk_with_ties(group, "p2p_rate_len", k)

        # Among the top k, select the row with the highest avg_yes_prob
        chosen_row = topk.loc[topk["avg_yes_prob"].idxmax()]

        final_corrects.append(chosen_row["gt_correct"])

    # Overall accuracy
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / N


def accuracy_topkp2p_then_yesprob_alt(df, k=3):
    df = df.copy()
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df["p2p_rate_len"] = df["p2p_rates"]

    # Helper function to get top-k rows with ties based on a given column.
    def get_topk_with_ties(group, col, k):
        group_sorted = group.sort_values(by=col, ascending=False)
        if len(group_sorted) <= k:
            return group_sorted
        kth_value = group_sorted.iloc[k - 1][col]
        return group_sorted[group_sorted[col] >= kth_value]

    grouped = df.groupby("docker_images")
    final_corrects = []

    for docker_image, group in grouped:
        # Normalize num_steps for the group.
        steps_min = group["num_steps"].min()
        steps_max = group["num_steps"].max()
        if steps_max == steps_min:
            group["num_steps_norm"] = 0.0
        else:
            group["num_steps_norm"] = (group["num_steps"] - steps_min) / (steps_max - steps_min)

        # Compute the adjusted score.
        group["adjusted_score"] = group["avg_yes_prob"]  # - group['num_steps_norm']

        # Select top k rows based on p2p_rate_len (keeping ties).
        topk = get_topk_with_ties(group, "p2p_rate_len", k)

        # Among these top rows, select the row with the highest adjusted_score.
        chosen_row = topk.loc[topk["adjusted_score"].idxmax()]
        final_corrects.append(chosen_row["gt_correct"])

    if not final_corrects:
        return 0.0
    return sum(final_corrects) / len(final_corrects)


def get_topk_with_ties(group, key, k, ascending=False):
    # Sort the group by 'p2p_rate_len' in descending order.
    group_sorted = group.sort_values(by=key, ascending=ascending)
    if len(group_sorted) <= k:
        return group_sorted
    kth_value = group_sorted.iloc[k - 1][key]
    # Keep all rows where p2p_rate_len is >= kth_value
    return group_sorted[group_sorted[key] >= kth_value]


def accuracy_topkp2p_then_exitreason_then_yesprob(df, k=3):
    df = df.copy()
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df["p2p_rate_len"] = df["p2p_rates"]

    grouped = df.groupby("docker_images")
    final_corrects = []

    for docker_image, group in grouped:
        # Sort by p2p_rate_len descending and take the top k rows
        # topk = group.sort_values(by="p2p_rate_len", ascending=False).head(k)
        topk = get_topk_with_ties(group, "p2p_rate_len", k)
        # import pdb; pdb.set_trace()
        # now topk 'agent' in exit_reasons (but only if there is at least one agent)
        if len(topk[topk["exit_reasons"] == "agent"]) > 0:
            topk = topk[topk["exit_reasons"] == "agent"]
        # if len(topk[topk["exit_reasons"].str.contains("agent")]) > 0:
        #     topk = topk[topk["exit_reasons"].str.contains("agent")]

        # Among the top k, select the row with the highest avg_yes_prob
        chosen_row = topk.loc[topk["avg_yes_prob"].idxmax()]

        final_corrects.append(chosen_row["gt_correct"])

    # Overall accuracy
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / N


def accuracy_topkexitreason_then_yesprob(df, k=3):
    df = df.copy()
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df["p2p_rate_len"] = df["p2p_rates"]

    grouped = df.groupby("docker_images")
    final_corrects = []

    for docker_image, group in grouped:
        # Sort by p2p_rate_len descending and take the top k rows
        # topk = group.sort_values(by="p2p_rate_len", ascending=False).head(k)
        topk = group  # get_topk_with_ties(group, "p2p_rate_len", k)
        # import pdb; pdb.set_trace()
        # now topk exit_reason=agent (but only if there is at least one agent)
        if len(topk[topk["exit_reasons"] == "agent"]) > 0:
            topk = topk[topk["exit_reasons"] == "agent"]

        # Among the top k, select the row with the highest avg_yes_prob
        chosen_row = topk.loc[topk["avg_yes_prob"].idxmax()]

        final_corrects.append(chosen_row["gt_correct"])

    # Overall accuracy
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / N


def accuracy_topkyesprob_then_exitreason_then_p2p(df, k=3):
    df = df.copy()
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df["p2p_rate_len"] = df["p2p_rates"]

    grouped = df.groupby("docker_images")
    final_corrects = []

    for docker_image, group in grouped:
        # Sort by avg_yes_prob descending and take top k rows
        topk = group.sort_values(by="avg_yes_prob", ascending=False).head(k)

        # now topk 'agent' in exit_reasons (but only if there is at least one agent)
        if len(topk[topk["exit_reasons"] == "agent"]) > 0:
            topk = topk[topk["exit_reasons"] == "agent"]
        # if len(topk[topk["exit_reasons"].str.contains("agent")]) > 0:
        #     topk = topk[topk["exit_reasons"].str.contains("agent")]

        # Among the top k, select the row with the highest p2p_rate_len
        chosen_row = topk.loc[topk["p2p_rate_len"].idxmax()]
        final_corrects.append(chosen_row["gt_correct"])

    # Overall accuracy
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / N


df = pd.read_csv("./deepswe-verifier-agent-priority-qwen3-14b-lora64_N16_traindata_p2p-14B-75k-lr1en5-v1.csv")

# load the regression data
USE_REGRESSION_DATA = True
if USE_REGRESSION_DATA:
    # load hf dataset with regression count (r2e-edits/deepswe-swebv-eval-n16-verifier-v1-with-regression)
    from datasets import load_dataset

    print("size of df before merging:", len(df))
    dataset = load_dataset("r2e-edits/deepswe-swebv-eval-n16-verifier-v1-with-regression", split="train")
    df_regression = dataset.to_pandas()
    # merge df_regression with df on docker_images and exp_names (i.e. both should match)
    df = pd.merge(df, df_regression, on=["docker_images", "exp_names"], how="left")
    print("size of df after merging:", len(df))
    # use regression_pass_count instead of p2p_rates
    df["p2p_rates"] = df["regression_pass_count"]
    df["exit_reasons"] = df["exit_reasons_x"]


def run_aggregation(df):
    # Group by 'docker_images' and select the row with the maximum avg_yes_prob from each group
    aggregated = df.groupby("docker_images", as_index=False).apply(lambda group: group.loc[group["avg_yes_prob"].idxmax()]).reset_index(drop=True)

    # Calculate the overall mean of 'gt_correct'
    overall_accuracy = aggregated["gt_correct"].mean()
    print("total number of trajs:", len(df))
    print("Overall Accuracy:", overall_accuracy)

    # if avg_yes_prob > 0.5, the predicted answer is YES (1), otherwise NO (0)
    df["predicted"] = (df["avg_yes_prob"] > 0.5).astype(int)
    df["gt_correct"] = df["gt_correct"].astype(int)
    df["correct_prediction"] = (df["predicted"] == df["gt_correct"]).astype(int)

    # Calculate the overall average accuracy
    overall_accuracy = df["correct_prediction"].mean()
    print("\nOverall RM Accuracy:", overall_accuracy)

    # defaults k=10 for first and k=1 for second
    k = 10
    acc_topk = accuracy_topkyesprob_then_p2p(df, k=k)
    print(f"Accuracy (top k [{k}] yesprob -> highest p2p_rate):", acc_topk)
    k = 5
    acc_topk = accuracy_topkp2p_then_yesprob(df, k=k)
    print(f"Accuracy (top k [{k}] p2p -> highest yesprob):", acc_topk)
    k = 5
    acc_topk = accuracy_topkp2p_then_exitreason_then_yesprob(df, k=k)
    print(f"Accuracy (top k [{k}] p2p -> exitreason=agent -> highest yesprob):", acc_topk)
    k = 5
    acc_topk = accuracy_topkexitreason_then_yesprob(df, k=k)
    print(f"Accuracy (top k [{k}] exitreason=agent -> highest yesprob):", acc_topk)
    k = 5
    acc_topk = accuracy_topkyesprob_then_exitreason_then_p2p(df, k=k)
    print(f"Accuracy (top k [{k}] yesprob -> exitreason=agent -> highest p2p_rate):", acc_topk)
    print("--------------------------------")

    k = 1
    for k in range(1, 11):
        print("k:", k)
        acc_topk = accuracy_topkyesprob_then_p2p(df, k=k)
        print(f"Accuracy (top k [{k}] yesprob -> highest p2p_rate):", acc_topk)
    print("--------------------------------")
    # loop over k=1,2,3,4,5, 6, 7, 8, 9, 10
    for k in range(1, 11):
        print("k:", k)
        acc_topk = accuracy_topkp2p_then_yesprob(df, k=k)
        print(f"Accuracy (top k [{k}] p2p -> highest yesprob):", acc_topk)
    print("--------------------------------")
    for k in range(1, 11):
        print("k:", k)
        acc_topk = accuracy_topkp2p_then_exitreason_then_yesprob(df, k=k)
        print(f"Accuracy (top k [{k}] p2p -> exitreason=agent -> highest yesprob):", acc_topk)
    print("--------------------------------")
    for k in range(1, 11):
        print("k:", k)
        acc_topk = accuracy_topkexitreason_then_yesprob(df, k=k)
        print(f"Accuracy (top k [{k}] exitreason=agent -> highest yesprob):", acc_topk)
    print("--------------------------------")
    for k in range(1, 11):
        print("k:", k)
        acc_topk = accuracy_topkyesprob_then_exitreason_then_p2p(df, k=k)
        print(f"Accuracy (top k [{k}] yesprob -> exitreason=agent -> highest p2p_rate):", acc_topk)
    print("--------------------------------")


run_aggregation(df)

# if __name__=='main':
#     main
