import json
import random
from collections import defaultdict

def extract_by_docker_image_upsample(
    all_jsonl, candicate_jsonl, out_jsonl,
    limit_min_num=-1, limit_max_turn=-1
):
    # Load candidate problem_statement set
    with open(candicate_jsonl, "r", encoding="utf-8") as f:
        wanted = {json.loads(line)["problem_statement"].strip() for line in f}
    print(len(wanted ))
    # Cluster by problem_statement (ps)
    clusters = defaultdict(list)
    count_all = 0
    with open(all_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            count_all+=1
            data = json.loads(line)
            ps = data["input"][2]["content"].split("</issue_description>")[0].split("<issue_description>")[-1].strip()
            if ps in wanted:
                clusters[ps].append(data)

    print(f"Data entries in the original file: {count_all}")

    total_upsampled = 0
    total_downsampled = 0
    upsample_stats = defaultdict(int)    # added count : number of ps
    downsample_stats = defaultdict(int)  # deleted count : number of ps
    all_write_num = 0
    # Write refined data
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for ps, items in clusters.items():
            count = len(items)

            # =============== Limit maximum number of turns (priority processing) ==================
            if limit_max_turn > -1 and count > limit_max_turn:
                # Sort by turns in ascending order
                items.sort(key=lambda x: len(x["input"]))
                deleted = count - limit_max_turn
                items = items[:limit_max_turn]
                clusters[ps] = items  # Update
                total_downsampled += deleted
                downsample_stats[deleted] += 1

            # ================== Upsample insufficient parts ==================
            count = len(items)
            if limit_min_num > -1 and count < limit_min_num:
                need = limit_min_num - count
                sampled = random.choices(items, k=need)
                items.extend(sampled)
                total_upsampled += need
                upsample_stats[need] += 1

            # Write to file
            for d in items:
                all_write_num+=1
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("======== Processing Statistics ========")
    print("Number of filtered PS:", len(clusters))
    print("Upsampling statistics:", dict(upsample_stats))
    print("Total upsampled count:", total_upsampled)
    print("Downsampling statistics:", dict(downsample_stats))
    print("Total deleted trajectory count:", total_downsampled)
    print("======== Final Results ========")
    print("Final trajectory count remaining:", all_write_num)

    return clusters, upsample_stats, downsample_stats, total_upsampled, total_downsampled


all_jsonl = "xx.jsonl"

candicate_jsonl = "./R2E-Gym/results/0_bon_filter_resultes/0121_swe_bon/bon_stats_1e-07_0.9999999.jsonl"

out_jsonl =  "yy.jsonl"


extract_by_docker_image_upsample(
    all_jsonl,
    candicate_jsonl,
    out_jsonl,
    limit_min_num=-2,  
    limit_max_turn=-2
)