import json

input_json_path = "your_input_file.json"  
output_json_path = "filtered_output.json"  

with open(input_json_path, 'r', encoding='utf-8') as f:
    data_list = json.load(f)


filtered_list = [
    item for item in data_list 
    if item["pre_difficulty"] not in (0, 1)
]


with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_list, f, ensure_ascii=False)