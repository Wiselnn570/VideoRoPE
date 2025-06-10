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

def tsv_to_json(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            data.append(row)
    return eval(json.dumps(data, indent=4))


def eval_dataset(args):
    min_pixels, max_pixels, context_length = args.min_pixels, args.max_pixels, args.context_length
    model_path = os.path.expanduser(args.model_path)
    if context_length >= 48000:
        llm = LLM(model_path,
                max_model_len=context_length+1536,
                limit_mm_per_prompt={"video": 10},
                gpu_memory_utilization=0.8
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
    qa_json = args.Eval_QA_root
    

    data = tsv_to_json(qa_json)
    keys = get_chunk(data, args.num_chunks, args.chunk_idx)

    answer_prompt = "\nAnswer with the option's letter from the given choices directly."

    eval_dict = {}
    
    eval_dataset_json = os.path.join(args.chat_conversation_output_folder, f"{str(args.chunk_idx)}.json")
    os.makedirs(os.path.dirname(eval_dataset_json), exist_ok=True)
    if os.path.exists(eval_dataset_json):
        st = set([item['index'] for item in read_jsonl(eval_dataset_json)])
    else:
        st = set()
    for v_id, _ in tqdm(enumerate(keys)):

        # 清理显存碎片
        torch.cuda.empty_cache()

        item = keys[v_id]
        if item['index'] in st: continue
        question = item['question']
        def adjust_qs_format(question):
            question = question.replace('\nA. ', '\nOptions:\n(A) ').replace('\nB. ', '\n(B) ').replace('\nC. ', '\n(C) ').replace('\nD. ', '\n(D) ')
            return question
        # question = adjust_qs_format(question)
        # question += answer_prompt
        video_path = os.path.join(args.Eval_Video_root, item['video_path'].split('/')[-1])
        
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
        try:
            image_inputs, video_inputs = process_vision_info(messages)
            print(len(video_inputs), video_inputs[0].shape)
        except:
            continue
        
        if context_length < 48000:
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

        pred = copy.deepcopy(_)
        pred.update({'prediction': generated_text})
        
        save_jsonl_line(eval_dataset_json, pred)

        # 删除无用变量并清理显存
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/hwfile/mllm/weixilin/cache/Qwen2-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--nframes", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--Eval_QA_root", type=str,
                        default='/mnt/hwfile/mllm/weixilin/VLMEvalKit/playground/data/videomme/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/Video-MME.tsv', help="folder containing QA JSON files")
    parser.add_argument("--Eval_Video_root", type=str,
                        default='/fs-computility/mllm/shared/dongxiaoyi/share_data/temp_data/Video-MME/videos', help="folder containing video data")
    parser.add_argument("--chat_conversation_output_folder",
                        type=str, default='/mnt/petrelfs/weixilin/projects/MLLM/Qwen2-VL/playground/results/', help="")
    parser.add_argument("--context_length", type=float, default=16384)
    parser.add_argument("--min_pixels", type=float, default=224 * 224)
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
