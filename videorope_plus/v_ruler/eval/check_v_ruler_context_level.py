import os
import json

def read_json(file_path):
    """
    从指定的路径读取 JSON 文件并返回数据。
    
    参数:
        file_path (str): JSON 文件的路径。
        
    返回:
        object: JSON 文件解析后的 Python 数据结构（字典或列表）。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data, file_path, indent=4):
    """
    将数据写入到指定路径的 JSON 文件。
    
    参数:
        data (dict 或 list): 需要写入的数据。
        file_path (str): JSON 文件的保存路径。
        indent (int, 可选): 格式化缩进的空格数，默认为 4。
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
def main(eval_dir):
    # eval_dir = '/fs-computility/mllm/weixilin/videorope_rebuttal/playground/results/longvideobench/v_ruler/Qwen2.5-VL-3B-Instruct-t_scale2_change_freq-128frames-16card_8k-context-330k-llava-video/default'
    for context_length in os.listdir(eval_dir):
        if '.json' in context_length:
            continue
        accuracy = {}
        context_dir = os.path.join(eval_dir, context_length)
        fg = False
        scores = []
        for depth_idx in os.listdir(context_dir):
            if '.json' in depth_idx: 
                fg = True
                break
            sta_path = os.path.join(context_dir, depth_idx, 'upload_board.json')
            if not os.path.exists(sta_path):
                fg = True
                break

            avg_score = read_json(sta_path)["Avg"]
            accuracy[depth_idx] = avg_score
            scores.append(avg_score)
        if fg:
            continue
        accuracy['Avg'] = sum(scores) / len(scores)
        numeric_items = sorted(
            ((k, v) for k, v in accuracy.items() if k != "Avg"),
            key=lambda x: float(x[0])
        )

        # 如果有 Avg，单独加到最后
        if "Avg" in accuracy:
            numeric_items.append(("Avg", accuracy["Avg"]))

        # 转为有序字典（Python 3.7+ 默认 dict 是有序的）
        ordered_data = dict(numeric_items)
        write_json(ordered_data, os.path.join(context_dir, 'upload_board.json'))
    
if __name__ == '__main__':
    import sys
    main(sys.argv[1])
