import numpy as np
import sys
import argparse

import json
import re
def read_json_file(file_path):
    """
    Reads a JSON file and returns the parsed data.

    :param file_path: str, path to the JSON file.
    :return: dict or list, parsed data from the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. Details: {e}")
        return None
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
def read_json_list(path_dir):
    import os
    lines = []
    for p in sorted(os.listdir(path_dir)):
        json_data = read_jsonl(os.path.join(path_dir, p))
        for data in json_data:
            if data['question_type'] == 'summary' or data['question_type'] == 'subPlot': continue
            data.update({'relative_dir': p.replace('.json', '')})
            lines.append(data)
    return lines


import os
def proxy_on():
    os.environ['http_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128'
    os.environ['https_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128'
    os.environ['HTTP_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128'
    os.environ['HTTPS_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128'
def proxy_off():
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''

def save_to_json(data, file_path, ensure_ascii=False, indent=4):
    """
    将数据保存为 JSON 文件。

    参数：
        data (dict or list): 要保存的数据（通常是字典或列表）。
        file_path (str): 保存的文件路径。
        ensure_ascii (bool): 是否强制以 ASCII 格式保存。默认为 False。
        indent (int): JSON 格式化的缩进级别。默认为 4。
    
    功能：
        - 自动创建目录（如果不存在）。
        - 保存数据为指定路径的 JSON 文件。
    """
    # 创建保存路径的目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 保存 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=ensure_ascii, indent=indent)
    # print(f"Data saved successfully to {file_path}")

def save_jsonl_line(data_entry, file_path):
    global lock
    """
    安全地将一条数据追加写入 JSONL 文件。

    参数:
    data_entry (dict): 要写入的 JSON 数据。
    file_path (str): JSONL 文件的路径。
    """
    # 使用线程锁来确保线程安全的写入
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")

def cal_score(results):
    basic_acc = 0
    halluc_acc = 0
    acc = 0
    for result in results:
        
        basic_hit = 0
        halluc_hit = 0
        final_hit = 0

        basic_answer = result["basic"]["answer"]
        basic_predict = result["basic"]["predict"]
        basic_answer_pattern = r'\b('+basic_answer+ r')\b'
        if re.search(basic_answer_pattern, basic_predict, re.IGNORECASE):
            basic_hit = 1

        halluc_answer = result["hallucination"]["answer"]
        halluc_predict = result["hallucination"]["predict"]
        halluc_answer_pattern = r'\b('+halluc_answer+ r')\b'
        if re.search(halluc_answer_pattern, halluc_predict, re.IGNORECASE):
            halluc_hit = 1
        
        final_hit = int(basic_hit and halluc_hit)

        basic_acc += basic_hit
        halluc_acc += halluc_hit
        acc += final_hit
    
    scores = {
        "basic_accuracy": basic_acc / len(results),
        "halluc_accuracy": halluc_acc / len(results),
        "accuracy": acc / len(results)
    }

    return scores

if __name__ == '__main__':
    import os

    path_root = sys.argv[1]
    final_result  = {}
    for tp in os.listdir(path_root):
        json_data = []
        for p in os.listdir(os.path.join(path_root, tp)):
            json_data += read_jsonl(os.path.join(path_root, tp, p))
        scores = cal_score(json_data)
        
        final_result[tp] = scores

    final_acc = 0
    final_basic_acc = 0
    final_halluc_acc = 0
    eval_type = ""
    for halluc_type, result in final_result.items():
        eval_type += halluc_type + "_"
        final_basic_acc += result["basic_accuracy"]
        final_halluc_acc += result["halluc_accuracy"]
        final_acc += result["accuracy"]
    if len(final_result.keys()) != 0:
        final_acc = final_acc / len(final_result.keys())
        final_basic_acc = final_basic_acc / len(final_result.keys())
        final_halluc_acc = final_halluc_acc / len(final_result.keys())
        final_result["all"] = {
            "basic_accuracy": final_basic_acc,
            "halluc_accuracy": final_halluc_acc,
            "accuracy": final_acc,
        }

        print("="*20)
        print("Basic Accuracy: ", final_basic_acc)
        print("Hallucination Accuracy: ", final_halluc_acc)
        print("Final Accuracy: ", final_acc)
        
        print("="*20)
    eval_result_path = os.path.join(path_root, 'upload_leaderboard.json')
    with open(eval_result_path, "w") as jp:
        json.dump(final_result, jp, indent=4)
