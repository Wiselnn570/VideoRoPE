import torch
from torchvision import transforms
import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
import json
import copy
import re
def check_ans(gt, pred):
    return extract_characters_regex(gt) == extract_characters_regex(pred)

def save_upload_json(save_dir, acc_dict):
    final_res = {}
    total_list = []
    for k, v in acc_dict.items():
        score_list = []
        for item in v:
            score_list.append(item['score'])
            total_list.append(item['score'])
        final_res[k] = sum(score_list) / len(score_list) * 100
    
    final_res['Avg'] = sum(total_list) / len(total_list) * 100
    save_to_json(final_res, os.path.join(save_dir, 'upload_board.json'))
    return final_res


def save_to_json(data, file_path, indent=4, ensure_ascii=False):
    """
    保存数据到 JSON 文件。

    参数:
        data (any): 要保存的数据，通常为字典或列表。
        file_path (str): 保存的文件路径，例如 'output.json'。
        indent (int): 缩进级别，默认为 4。
        ensure_ascii (bool): 是否确保 ASCII 编码，默认为 False (支持非 ASCII 字符)。
    
    返回:
        None
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=indent, ensure_ascii=ensure_ascii)
        print(f"数据已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存到 JSON 文件时发生错误: {e}")

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        return ''
    return matches[0]
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
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
    
def read_json_list(path_dir):
    import os
    lines = []
    import re
    for p in sorted(os.listdir(path_dir)):
        if re.match(r'^\d+\.json$', p):
            json_data = read_jsonl(os.path.join(path_dir, p))
            for data in json_data:
                data.update({'relative_dir': p.replace('.json', '')})
                lines.append(data)
    return lines

def main(eval_dir):
    
    data_all = read_json_list(eval_dir)
    assert len(data_all) == 1337 or len(data_all) == 964, f'data not reach 1337 or 964, but got {len(data_all)}'
    print(data_all[0])
    acc_dict = {}
    for data in data_all:
        tp = data['question_category']
        score = int(check_ans(data['answer'], data['prediction']))
        data.update({'score': score})
        if tp not in acc_dict:
            acc_dict[tp] = [copy.deepcopy(data)]
        else:
            acc_dict[tp].append(copy.deepcopy(data))
    save_upload_json(eval_dir, acc_dict)
    save_to_json(acc_dict, os.path.join(eval_dir, 'results.json'))

if __name__ == '__main__':
    import sys
    main(sys.argv[1])