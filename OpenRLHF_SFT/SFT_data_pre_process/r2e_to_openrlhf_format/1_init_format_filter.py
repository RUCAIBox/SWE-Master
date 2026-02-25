import json
from collections import defaultdict
hint_content = "You forgot to use a function call in your response. "
in_file = "./R2E-Gym/results/swe_all_w_sp_obs.jsonl"
out_file = "./R2E-Gym/results/swe_all_w_sp_obs_filter.jsonl"
failed_num = defaultdict(int)
success_num = 0

with open(in_file, "r", encoding="utf-8") as fin, \
     open(out_file, "w", encoding="utf-8") as fout:

    for line in fin:
        skip_flag=False
        sample = json.loads(line)
        if sample["reward"]!=1:
            continue
        inputs = sample["input"]
        
        if "<function=></function>" in inputs[-4]["content"].replace("\n","").replace(" ",""):
            # print(inputs[-4]["content"][-100:])
            inputs[-4]["content"] = inputs[-4]["content"].replace("<function=>","<function=submit>")
            inputs = inputs[:-2]

        all_str = ""
        for mess in inputs[:-1]:
            all_str += mess["content"]
        step_count = sample["step_count"]
        token_usage_total =sample["token_usage_total"]
        
        # print(inputs[-2]["content"][-100:])
        # print("=="*50)
        if (hint_content in all_str) : 
            # print(inputs[-2]["content"][-100:])
            # print("=="*50)
            # if hint_content not in inputs[-4]["content"]
            failed_num["have_hint_content"]+=1
            skip_flag=True
        # print(all_str)
        # exit()

        if step_count == 100 or step_count == 120 or step_count == 150:
            failed_num["max_step_limit"]+=1
            skip_flag=True
        # if token_usage_total >=98304:
        # if token_usage_total >=65536:
        # if token_usage_total >=40960:
        # if token_usage_total >=32768:
        if token_usage_total >=81920 :
            failed_num["max_token_limit"]+=1
            skip_flag=True
        if len(inputs) < 2:
            failed_num["so_less_turn"]+=1
            skip_flag=True

        if "<function=submit>" not in inputs[-2]["content"]:
            failed_num["final_rool_not_submit"]+=1
            skip_flag=True
            # print(inputs[-2]["content"])
            # print(len(inputs))
            if "<function=>" in inputs[-2]["content"]:
                pass
                # print(inputs[-1]["content"])
                # print(len(inputs))
                # print("=="*50)
            else:
                pass
                # failed_num["final_rool_not_submit"]+=1
                # skip_flag=True

        final_mess = inputs[-2]
        if ("<function=submit>\n</function>" not in final_mess["content"]) and  ("<function=submit></function>" not in final_mess["content"]) :
            failed_num["submit_with_paras"]+=1
            skip_flag=True
            
        
        if inputs[-1]["role"]!= "tool":
            failed_num["final_role_not_tool"]+=1
            skip_flag=True

        if skip_flag:
            continue

        # if "Finish" not in inputs[-1]["content"]:
        #     print(inputs)
        #     kill
        sample["input"] = inputs[:-1]
        # print(inputs[-1])
        success_num+=1
        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

print("Finished →", out_file)
print(f"Count of failed/excluded entries: {failed_num}")
print(f"Count of successful/kept entries: {success_num}")

