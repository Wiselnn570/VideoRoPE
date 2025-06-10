import sys

import argparse
import json
import math
import os
import tempfile
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
import requests
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import tempfile
import copy
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import random

def setup_seed(seed=428):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

rel_path = {
    'obj_rel': {
        'json': "object_relation/object_relation.json",
        'video': "object_relation/videos",
    },
    'temporal': {
        'json': "temporal/temporal.json",
        'video': "temporal/videos",
    },
    'semantic': {
        'json': "semantic_detail/semantic_detail.json",
        'video': "semantic_detail/videos",
    },
    'fact': {
        'json': "external_factual/external_factual.json",
        'video': "external_factual/videos",
    },
    'nonfact': {
        'json': "external_nonfactual/external_nonfactual.json",
        'video': "external_nonfactual/videos",
    },
    'factdet': {
        'json': "fact_detect/fact_detect.json",
        'video': "fact_detect/videos"
    }
}

import json

def save_jsonl_line(file_path, data):
    """
    将单个数据行以 JSON Lines 格式保存到文件中。

    参数：
    file_path (str): JSONL 文件的路径。
    data (dict): 需要保存的数据，必须是可序列化为 JSON 的字典或列表。
    """
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data) + '\n')

def proxy_off():
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''

def proxy_on():
    os.environ['http_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128'
    os.environ['https_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128'
    os.environ['HTTP_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128'
    os.environ['HTTPS_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128'

proxy_off()

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    import random
    random.seed(233)
    random.shuffle(lst)
    chunks = split_list(lst, n)
    return chunks[k]


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq

import json
import csv
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
def tsv_to_json(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            data.append(row)
    return eval(json.dumps(data, indent=4))

def read_json_list(path_dir):
    import os
    lines = []
    for p in sorted(os.listdir(path_dir)):
        json_data = read_json_file(os.path.join(path_dir, p))
        for data in json_data:
            if data['question_type'] == 'summary' or data['question_type'] == 'subPlot': continue
            data.update({'relative_dir': p.replace('.json', '')})
            lines.append(data)
    return lines



def eval_dataset(args):
    min_pixels, max_pixels, context_length = args.min_pixels, args.max_pixels, args.context_length
    model_path = os.path.expanduser(args.model_path)
    if context_length > 48000:
        llm = LLM(model_path,
                max_model_len=context_length+1536,
                limit_mm_per_prompt={"video": 10},
                )
        total_pixels = (context_length-512) * 28 * 28
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, 
                                                            device_map="cpu",
                                                            torch_dtype=torch.bfloat16, 
                                                            attn_implementation="flash_attention_2"
                                                            )
        model = model.to('cuda')
        model = model.eval()
        total_pixels = (context_length-512) * 28 * 28

    if args.nframes is None:
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, total_pixels=total_pixels)
    else:
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=min_pixels, nframes=args.nframes)
    path_root = args.Eval_root

    qa_path = os.path.join(path_root, rel_path[args.qa_type]['json'])
    video_dir_path = os.path.join(path_root, rel_path[args.qa_type]['video'])
    setup_seed(42)


    data = read_json_file(qa_path)
    keys = get_chunk(data, args.num_chunks, args.chunk_idx)
    answer_prompt = "\nAnswer the question using 'yes' or 'no'."

    eval_dict = {}
    
    eval_dataset_json = os.path.join(args.chat_conversation_output_folder, f"{str(args.chunk_idx)}.jsonl")
    os.makedirs(os.path.dirname(eval_dataset_json), exist_ok=True)
    # import pdb; pdb.set_trace()
    if os.path.exists(eval_dataset_json):
        st = set([item['basic']['video']+item['basic']['question']+item['basic']['answer']+item['type']+args.qa_type for item in read_jsonl(eval_dataset_json)])
    else:
        st = set()
    
    for v_id, _ in tqdm(enumerate(keys)):

        # 清理显存碎片
        torch.cuda.empty_cache()

        item = keys[v_id]
        if item['basic']['video']+item['basic']['question']+item['basic']['answer']+item['type']+args.qa_type in st: continue
        
        for qs_type in ['basic', 'hallucination']:
            # import pdb; pdb.set_trace()
            video_path = os.path.join(video_dir_path, item[qs_type]['video'])
            question = item[qs_type]['question'] + answer_prompt
        
            if args.nframes is None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "min_pixels": min_pixels,
                                # "max_pixels": max_pixels,
                                "total_pixels": total_pixels,
                                # "nframes": args.nframes,
                            },
                            {"type": "text", "text": question},
                        ],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "min_pixels": min_pixels,
                                # "max_pixels": max_pixels,
                                "max_pixels": min_pixels,
                                "nframes": args.nframes,
                            },
                            {"type": "text", "text": question},
                        ],
                    },
                ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            print(video_inputs[0].shape[0] // 2 * video_inputs[0].shape[2] // 28 * video_inputs[0].shape[3] // 28)
            # import pdb; pdb.set_trace()
            if context_length <= 48000:
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                inputs = inputs.to(model.device)
                
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, which_rope=args.which_rope, scale_factor=args.scale_factor)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    generated_text = output_text[0]
                print(generated_text)
            else:
                h_bar, w_bar = video_inputs[0].shape[2], video_inputs[0].shape[3]
                compute_context = video_inputs[0].shape[0] // 2 * video_inputs[0].shape[2] // 28 * video_inputs[0].shape[3] // 28
                if compute_context > context_length:
                    scal = round(context_length / compute_context, 2)
                    resized_height, resized_width = int(scal * h_bar), int(scal * w_bar)
                    def adjust_to_multiple_of_28(value):
                        return ((value + 27) // 28) * 28
                    if adjust_to_multiple_of_28(resized_height) == h_bar and adjust_to_multiple_of_28(resized_width) == w_bar:
                        resized_height, resized_width = (h_bar // 28 - 1) * 28, (w_bar // 28 - 1) * 28
                    video_inputs[0] = video_inputs[0].byte()
                    video = transforms.functional.resize(
                        video_inputs[0],
                        [resized_height, resized_width],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ).float()
                    video_inputs = [video]
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                mm_data['which_rope'] = args.which_rope
                mm_data['scale_factor'] = args.scale_factor
                llm_inputs = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                }
                with torch.no_grad():
                    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
                generated_text = outputs[0].outputs[0].text
                print(generated_text)
                del mm_data, llm_inputs, outputs

            item[qs_type]["predict"] = generated_text

            # 删除无用变量并清理显存
            del messages, prompt, image_inputs, video_inputs
            torch.cuda.empty_cache()
        item['qa_type'] = args.qa_type
        save_jsonl_line(eval_dataset_json, item)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/hwfile/mllm/weixilin/cache/Qwen2-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--nframes", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--chunk-idx", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--Eval_root", type=str,
                        default='/mnt/hwfile/mllm/weixilin/VideoHallucer/videohallucer_datasets/snapshots/71d7ed2153be6e0c9383d7d142b75feea7a3e8e9', help="folder containing QA JSON files")
    parser.add_argument("--chat_conversation_output_folder",
                        type=str, default='/mnt/petrelfs/weixilin/projects/MLLM/Qwen2-VL/playground/results/videohallucer/Qwen2-VL-7B-Instruct-64000-144tokens', help="")
    parser.add_argument("--qa_type", type=str, choices=['obj_rel', 'temporal', 'semantic', 'fact', 'nonfact', 'factdet'], default='obj_rel')
    parser.add_argument("--context_length", type=float, default=32768)
    parser.add_argument("--min_pixels", type=float, default=144 * 28 * 28)
    parser.add_argument("--max_pixels", type=float, default=256 * 28 * 28)
    parser.add_argument("--which_rope", type=str, default='m_rope')
    parser.add_argument("--scale_factor", type=float, default=1.0)
    args = parser.parse_args()
    sampling_params = SamplingParams(
        best_of=1,
        temperature=0.0,
        top_p=1,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        presence_penalty=0,
        frequency_penalty=0,
    )

    eval_dataset(args)
