import os
import json

from collections import defaultdict

def process_folders_bon(folder_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Output file paths
    bon_output = os.path.join(output_dir, "bon_stats.jsonl")
    acc_stat_output = os.path.join(output_dir, "overall_acc_stats.jsonl")

    # key: docker_image -> reward list
    docker_stats = defaultdict(list)

    # Iterate through folders
    all_folder_num = len(folder_list)
    current_num = 0
    for folder_path in folder_list:
        current_num+=1
        jsonl_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".jsonl") and os.path.isfile(os.path.join(folder_path, f))
        ]

        for jsonl_file in jsonl_files:
            
            file_path = os.path.join(folder_path, jsonl_file)
            print(f"Current progress: {current_num}/{all_folder_num}; Starting to process file: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    # print(data["ds"].keys())
                    # exit()
                    # Do not use .get()
                    if "docker_image" not in data:
                        kill
                    if "reward" not in data:
                        kill

                    # docker_img = data["docker_image"]
                    docker_img = data["problem_statement"]
                    # print(data["ds"].keys())
                    # if 'commit_hash' in data["ds"]:
                    #     instance_id = data["ds"]["commit_hash"]
                    # else:
                    #     instance_id = data["ds"]["base_commit"]
                    instance_id = data["ds"]["instance_id"]
                    # try:
                    #     
                    #     print(1111)
                    # except:
                    #     print(data["ds"].keys())
                    #     print(data.keys())
                    #     exit()
                    problem_statement =data["problem_statement"]
                    # print( docker_img)
                    reward = data["reward"]

                    docker_stats[docker_img].append((reward, problem_statement,instance_id))

    # ========== Write docker_image level BON statistics ==========

    with open(bon_output, "w", encoding="utf-8") as out:
        for docker_img, reward_info in docker_stats.items():
            reward_list = [r for r, _, _ in reward_info]
            problem_statement = reward_info[0][1]  # One-to-one mapping with docker_image
            instance_id = reward_info[0][2]
            reward_mean = sum(reward_list) / len(reward_list)
            # print(reward_list)
            entry = {
                "docker_image": docker_img,
                "problem_statement":problem_statement,
                "reward_mean": reward_mean,
                "reward_list": reward_list,
                "count": len(reward_list),
                "instance_id":instance_id
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ========== Calculate overall acc -> sample count statistics ==========

    acc_count = {}  # key: mean acc, value: number of docker_image

    for docker_img, reward_info in docker_stats.items():
        reward_list = [r for r, _ ,_ in reward_info]
        mean_acc = sum(reward_list) / len(reward_list)
        if mean_acc in acc_count:
            acc_count[mean_acc] = acc_count[mean_acc] + 1
        else:
            acc_count[mean_acc] = 1

    # Sorting
    sorted_acc = sorted(acc_count.items(), key=lambda x: x[0], reverse=True)

    # ========== Write overall statistics file ==========
    with open(acc_stat_output, "w", encoding="utf-8") as out:
        for acc, num in sorted_acc:
            entry = {
                "acc": acc,
                "sample_num": num
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Print to console
    print("===== Average ACC Distribution (High to Low) =====")
    for acc, num in sorted_acc:
        print(f"ACC={acc:.4f}  ->  {num} samples")

    print("\nBON statistics completed!")
    print(f"docker_image level statistics written to: {bon_output}")
    print(f"Overall ACC distribution statistics written to: {acc_stat_output}")


import os
def find_subfolders_with_str(folder_path: str, keyword: str):
    result = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if os.path.isdir(full_path) and keyword in name:
            result.append(full_path)
    return result



folder_list = ["./R2E-Gym/results/1214_0.0001_0.4", "./R2E-Gym/results/1214_0.4_0.6"]


print("Carefully check the files to be processed!!!!!!")
for j in folder_list:
    print(j)
print("=="*40)
print(f"Number of files to be processed: {len(folder_list)}")
print("=="*40)

# exit()

output_dir = './R2E-Gym/results/0_bon_filter_resultes/0221_swe_bon'
process_folders_bon(folder_list, output_dir)
print(f"Total number of processed files: {len(folder_list)}")