import json

def filter_by_reward_range(input_jsonl, output_jsonl, min_acc, max_acc):
    with open(input_jsonl, "r", encoding="utf-8") as f, \
         open(output_jsonl, "w", encoding="utf-8") as out:
        num_a = 0
        for line in f:
            data = json.loads(line)

            if "reward_mean" not in data:
                continue

            acc = data["reward_mean"]

            if acc >= min_acc and acc < max_acc:
                if data["count"] >=1:
                    num_a+=1
                    out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Finished: {output_jsonl}")
    print(f"After Filtering, the num: {num_a}")

def filter_by_count(input_jsonl, output_jsonl, count):
    num=0
    with open(input_jsonl, "r", encoding="utf-8") as f, \
         open(output_jsonl, "w", encoding="utf-8") as out:

        for line in f:
            data = json.loads(line)

            count_rollout = data["count"]

            if count_rollout==count:
                num+=1
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Finished: {output_jsonl}")
    print(f"After Filtering, the num: {num}")


input_jsonl = "./R2E-Gym/results/0_bon_filter_resultes/0121_swe_bon/bon_stats.jsonl"

minacc = 0.0000001
maxacc = 0.7999999

minacc = 0.7999999
maxacc = 0.9999999

minacc = 0.9999999
maxacc = 1.0000001

minacc = 0.0000001
maxacc = 0.9999999

minacc = 0
maxacc = 0.0000001

minacc = 0.0000001
maxacc = 0.6000001

minacc = 0.0000001
maxacc = 0.4000001

minacc = 0.4000001
maxacc = 0.6000001

minacc = 0.3999999
maxacc = 0.9999999

minacc = 0.0000001
maxacc = 0.9999999

# minacc = 0.9999999
# maxacc = 1.0000001

# minacc = 0.600000
# maxacc = 0.999999

file_name = input_jsonl.split(".jsonl")[0]
output_jsonl = f"{file_name}_{minacc}_{maxacc}.jsonl"
filter_by_reward_range(
    input_jsonl=input_jsonl,
    output_jsonl=output_jsonl,
    min_acc= minacc,
    max_acc= maxacc
)