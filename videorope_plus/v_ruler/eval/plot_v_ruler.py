import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from pathlib import Path
import random
import json

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
    all_accuries = []
    for context_length in sorted(
        [p for p in os.listdir(eval_dir) if '.' not in p],
        key=lambda x: int(x)
        ):
        if '.' in context_length: continue
        print(context_length)
        accuracy = {}
        context_dir = os.path.join(eval_dir, context_length)
        context_path = os.path.join(context_dir, 'upload_board.json')
        if not os.path.exists(context_path):
            break
        context_data = read_json(context_path)
        for k, v in context_data.items():
            if k == 'Avg':
                continue
            result = {
                'Context Length': int(context_length),
                "Frame Depth": round(float(k) * 100, -1),
                "Score": float(v) / 100
            }
            all_accuries.append(result)


    plot_path = eval_dir

    df = pd.DataFrame(all_accuries)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#9ad5b3"]
    )

    pivot_table = pd.pivot_table(
        df,
        values="Score",
        index=["Frame Depth", "Context Length"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(
        index="Frame Depth", columns="Context Length", values="Score"
    )
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        vmin=0,
        vmax=1,
        linecolor='white',
        linewidths=1.5, 
        cmap=cmap,
        cbar_kws={"label": "Score"},
    )

    # Set the color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=14)


    # Define the formatter function
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:.1f}K'
        return f'{x}'

    context_lengths = pivot_table.columns
    formatted_context_lengths = [thousands_formatter(x, None) for x in context_lengths]

    # More aesthetics
    plt.xlabel("Context Length", fontsize=14)  # X-axis label
    plt.ylabel("Depth Percent", fontsize=14)  # Y-axis label
    plt.xticks(ticks=[i + 0.5 for i in range(len(context_lengths))], labels=formatted_context_lengths, rotation=45, fontsize=14)
    # plt.xticks(rotation=45, fontsize=14)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=14)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    # model_name = args.model.split("/")[-1]


    plt.savefig(os.path.join(plot_path,"heatmap.png"))
    # calculate average accuracy
    average_accuracy = df["Score"].mean()
    print(f"Average Accuracy: {average_accuracy}")
    # save as txt
    with open(os.path.join(plot_path, "avg_accuracy.txt"), "w") as f:
        f.write(f"Average Accuracy: {average_accuracy}\n")

if __name__ == '__main__':
    import sys
    main(sys.argv[1])