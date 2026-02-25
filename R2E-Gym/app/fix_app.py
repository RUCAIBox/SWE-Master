import json
import os
import pandas as pd
import re
import html

def parquet_to_html(parquet_paths, suffixes, output_html):
    assert len(parquet_paths) == len(suffixes), "The number of parquet_paths and suffixes must be consistent"

    # Read all files
    all_dfs = [pd.read_parquet(p) for p in parquet_paths]

    # Cluster by prompt
    grouped = {}
    for df, suffix in zip(all_dfs, suffixes):

        for sample in df.to_dict(orient="records"):
            question_user = sample["prompt"].split("<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>user")
            if len(question_user) == 2:
                question = question_user[-1].split("<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant")[0].strip()
            elif len(question_user) == 3:
                question = question_user[1].split("<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant")[0].strip()
            else:
                question = ""

            if question not in bc_en_question:
                continue

            print(question)
            print("=="*50)
            prompt = sample["prompt"]
            gen = sample["gen"]
            score = sample["score"]

            if question not in grouped:
                grouped[question] = {}

            grouped[question][suffix] = {
                "prompt":prompt,
                "gen": gen,
                "score": score
            }
        print(len(grouped))

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>Parquet Comparison Visualization</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 20px; }",
        "    details { margin-left: 20px; margin-bottom: 10px; }",
        "    summary { cursor: pointer; font-weight: bold; }",
        "    pre { background: #f4f4f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; }",
        "    ul { list-style-type: none; padding-left: 20px; }",
        "    li { margin-bottom: 10px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Parquet Comparison Visualization</h1>"
    ]

    for idx, (question, gens) in enumerate(grouped.items(), 1):
        html_parts.append("<details>")
        html_parts.append(f"<summary>Question {idx}</summary>")

        html_parts.append("<h3>Question:</h3>")
        html_parts.append(f"<pre>{question}</pre>")

        # Side-by-side comparison layout
        html_parts.append("<div style='display: flex; gap: 20px;'>")

        for suffix in suffixes:
            prompt = gens.get(suffix, {}).get("prompt", "")
            # gen = gens.get(suffix, "")
            # chat_string = prompt + gen
            gen_info = gens.get(suffix, {})
            prompt = gen_info.get("prompt", "")
            gen = gen_info.get("gen", "")
            score = gen_info.get("score", "N/A")
            chat_string = prompt + gen
            # Assume parse_chatml_to_messages_standalone exists
            # messages = parse_chatml_to_messages_standalone(chat_string)[3:] 
            messages = [{"role": "placeholder", "content": "Messages would go here"}] # Placeholder

            html_parts.append(
                "<div style='flex: 1; width: 48%; max-height: 90vh; overflow-y: auto; "
                "border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
            )
            html_parts.append(f"<h4>{suffix} (score={score})</h4>")
            html_parts.append("<details>")
            html_parts.append(f"<summary>Show Messages</summary>")
            html_parts.append("<ul>")
            for m in messages:
                role = m.get("role", "")
                name = m.get("name", "")
                content = m.get("content", "")
                safe_content = html.escape(content)

                display_name = f" ({name})" if name else ""
                html_parts.append(f"<li><b>{role}{display_name}:</b> <pre>{safe_content}</pre></li>")

            html_parts.append("</ul>")
            html_parts.append("</details>")
            html_parts.append("</div>")

        html_parts.append("</div>")  # End flex container
        html_parts.append("</details>")


    html_parts.append("</body></html>")

    with open(output_html, "w") as f:
        f.write("\n".join(html_parts))

    print(f"HTML saved to {output_html}")


def postprocess_r2egym_traj(jsonl_f):
    messages_all = {}
    with open(jsonl_f,"r") as f:
        for line in f:
            messages = []
            data=json.loads(line)
            # print(data["problem_statement"])
            # print_dict(data)
            agent_args = data["agent_args"]
            trajectory_steps = data["trajectory_steps"]
            problem_statement = data["problem_statement"]
            # print_dict(agent_args)
            sp_1 = agent_args["system_prompt"]
            sp_2 = agent_args["instance_prompt"]
            sp_2 = sp_2.replace("{problem_statement}",problem_statement).replace("{working_dir}","/testbed")
            messages.append({"role":"system","content":sp_1})
            
            messages.append({"role":"user","content":sp_2})
            for step in trajectory_steps:
                thought = step["thought"]
                action = step["action"]
                observation = step["observation"]
                messages.append({"role":"assistant","content":thought+"\n\n"+action})
                messages.append({"role":"tool","content":observation})

            meta_info = {
                    "issue_name":data["env_args"]['ds']["instance_id"],
                    "exp_name":data["exp_name"],
                    "docker_image":data["docker_image"],}

            traj_info = {
                    "exit_reason":data["exit_reason"],
                    "used_iterations":(len(messages)-2)/2,
                    "reward":data['reward'] ,
                    "max_iterations":data["max_steps_absolute"],
                    "max_tokens":data["max_token_limit"],
                    }
            res_dict = {"messages":messages, "meta_info":meta_info,"traj_info":traj_info}
            messages_all[meta_info["issue_name"]] = res_dict
    return messages_all

def postprocess_openhands_traj(json_f_oh):
    with open(json_f_oh, 'r', encoding='utf-8') as file:
        data_all = json.load(file)
        messages_all ={}
        for data in data_all:
            messages = []
            instance_id = data["instance_id"]
            exit_reason = "TODO"
            reward = "TODO"
            max_iterations = "TODO" #data["metadata"]['max_iterations']
            max_tokens = "TODO"
            history = data["history"]
            sp_1 = history[0]["args"]["content"]
            sp_tool_1= json.dumps(history[0]["args"]["tools"])
            user_prompt = history[1]["args"]["content"]
            messages.append({"role":"system","content":sp_1})
            messages.append({"role":"system","content":sp_tool_1})
            messages.append({"role":"user","content":user_prompt})

            for his in history[4:]:
                message = his["message"]
                if "action" in his:
                    role = "assistant"
                    assert len(his["tool_call_metadata"]["model_response"]["choices"])==1
                    reasoning_content = his["tool_call_metadata"]["model_response"]["choices"][0]["message"]["content"]
                    tool_calls = json.dumps(his["tool_call_metadata"]["model_response"]["choices"][0]["message"]["tool_calls"])
                    if not reasoning_content:
                        reasoning_content = ""
                    else:
                        reasoning_content = reasoning_content.strip() + "\n\n"
                    messages.append({"role":"assistant","content": reasoning_content + tool_calls.strip() })
                elif "observation" in his:
                    role = "tool"
                    tool_name = his["tool_call_metadata"]["function_name"]
                    tool_response = his["content"]
                    messages.append({"role":"tool","content":tool_response.strip() })

                else:
                    raise ValueError("Invalid Role Type!!!!")
            meta_info = {
                    "issue_name":instance_id,
                    }
            traj_info = {
                    "exit_reason":exit_reason,
                    "used_iterations":(len(messages)-2)/2,
                    "reward":reward ,
                    "max_iterations":max_iterations,
                    "max_tokens":max_tokens,
                    }

            res_dict = {"messages":messages,"traj_info":traj_info,"meta_info":meta_info}
            messages_all[instance_id] = res_dict
        return messages_all

def dual_dicts_to_html(dict_left, suffix_left,
                       dict_right, suffix_right,
                       output_html):
    """
    Aligns two dicts (key=instance_id, value={"messages":..., "score":...}) by key to generate 
    a side-by-side comparison HTML. If an instance_id exists only on one side, 
    the other side will be displayed as empty.
    """

    all_ids = sorted(set(dict_left.keys()) | set(dict_right.keys()))

    # HTML Header
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>Dual Dict Comparison</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 20px; }",
        "    details { margin-left: 20px; margin-bottom: 10px; }",
        "    summary { cursor: pointer; font-weight: bold; }",
        "    pre { background: #f4f4f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; }",
        "    ul { list-style-type: none; padding-left: 20px; }",
        "    li { margin-bottom: 10px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Dual Dict Comparison</h1>"
    ]

    # Iterate through all instance_ids
    for idx, inst_id in enumerate(all_ids, 1):
        html_parts.append("<details>")
        elem_l = dict_left.get(inst_id)
        elem_r = dict_right.get(inst_id)
        
        # Safely retrieve the score
        s_l = elem_l["traj_info"]["reward"] if elem_l else "N/A"
        s_r = elem_r["traj_info"]["reward"] if elem_r else "N/A"

        html_parts.append(f"<summary>{inst_id} : {s_l } v.s. {s_r }</summary>")
        # html_parts.append("<h3>Question:</h3>")
        
        html_parts.append(f"<pre>{html.escape(inst_id)} </pre>")

        html_parts.append("<div style='display: flex; gap: 20px;'>")

        # Left side
        html_parts.append(
            "<div style='flex: 1; width: 48%; max-height: 90vh; overflow-y: auto; "
            "border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
        )
        if elem_l:
            score = elem_l["traj_info"]["reward"]
            iter_nums = elem_l["traj_info"]["used_iterations"]
            html_parts.append(f"<h4>{suffix_left} (score={score}) (iter_nums={iter_nums})</h4>")
            html_parts.append("<details><summary>Show Messages</summary><ul>")
            for m in elem_l["messages"]:
                role = m.get("role", "")
                name = m.get("name", "")
                content = html.escape(m.get("content", ""))
                display_name = f" ({name})" if name else ""
                html_parts.append(f"<li><b>{role}{display_name}:</b> <pre>{content}</pre></li>")
            html_parts.append("</ul></details>")
        else:
            html_parts.append(f"<h4>{suffix_left} (N/A)</h4>")
            html_parts.append("<li><i>Empty</i></li>")
        html_parts.append("</div>")


        # Right side
        html_parts.append(
            "<div style='flex: 1; width: 48%; max-height: 90vh; overflow-y: auto; "
            "border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
        )
        if elem_r:
            score = elem_r["traj_info"]["reward"]
            iter_nums = elem_r["traj_info"]["used_iterations"]
            html_parts.append(f"<h4>{suffix_right} (score={score}) (iter_nums={iter_nums})</h4>")
            html_parts.append("<details><summary>Show Messages</summary><ul>")
            for m in elem_r["messages"]:
                role = m.get("role", "")
                name = m.get("name", "")
                content = html.escape(m.get("content", ""))
                display_name = f" ({name})" if name else ""
                html_parts.append(f"<li><b>{role}{display_name}:</b> <pre>{content}</pre></li>")
            html_parts.append("</ul></details>")
        else:
            html_parts.append(f"<h4>{suffix_right} (N/A)</h4>")
            html_parts.append("<li><i>Empty</i></li>")
        html_parts.append("</div>")


        html_parts.append("</div></details>")

    html_parts.append("</body></html>")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"HTML saved to {output_html}")


# ----------------------------------------------------------------------------
# ⭐️ New function: used to display a single trajectory
# ----------------------------------------------------------------------------

def single_dict_to_html(traj_dict, suffix, output_html):
    """
    Generates a single-column HTML from a single dict (key=instance_id, value={"messages":..., "traj_info":...}).
    """
    
    all_ids = sorted(traj_dict.keys())

    # HTML Header
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>Single Trajectory Visualization</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 20px; }",
        "    details { margin-left: 20px; margin-bottom: 10px; border: 1px solid #eee; padding: 5px; border-radius: 5px; }",
        "    summary { cursor: pointer; font-weight: bold; }",
        "    pre { background: #f4f4f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; }",
        "    ul { list-style-type: none; padding-left: 20px; }",
        "    li { margin-bottom: 10px; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>Trajectory Visualization: {html.escape(suffix)}</h1>"
    ]

    # Iterate through all instance_ids
    for idx, inst_id in enumerate(all_ids, 1):
        elem = traj_dict.get(inst_id)
        if not elem:
            continue

        score = elem["traj_info"]["reward"]
        iter_nums = elem["traj_info"]["used_iterations"]
        
        html_parts.append("<details>")
        html_parts.append(f"<summary>{inst_id} (score={score}, iter_nums={iter_nums})</summary>")

        # Single column layout
        html_parts.append(
            "<div style='width: 95%; max-height: 90vh; overflow-y: auto; "
            "border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
        )
        
        html_parts.append("<details open><summary>Show Messages</summary><ul>")
        for m in elem["messages"]:
            role = m.get("role", "")
            name = m.get("name", "")
            content = html.escape(m.get("content", ""))
            display_name = f" ({name})" if name else ""
            html_parts.append(f"<li><b>{role}{display_name}:</b> <pre>{content}</pre></li>")
        
        html_parts.append("</ul></details>")
        html_parts.append("</div>") # End div
        html_parts.append("</details>") # End main details

    html_parts.append("</body></html>")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"HTML saved to {output_html}")


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Assume your file paths
    # Note: In your original code, the same file was loaded twice; this logic is maintained.
    traj_path = "./R2E-Gym/results/glm-swesmith-front-ready/swesmith_front-ready.jsonl"

    print("Processing smith...")
    r2e_gym_traj_cdx = postprocess_r2egym_traj(traj_path)
    print(f"Loaded {len(r2e_gym_traj_cdx)} trajectories for cdx.")
    

    name = traj_path.split("/")[-1].split(".jsonl")[0]
    
    output_path_single = f"./R2E-Gym/app/results/{name}.html"
    
    single_dict_to_html(
        r2e_gym_traj_cdx, 
        name, 
        output_path_single
    )
    print("Single trajectory view HTML generated.")

