# # Set VLLM_WORKER_MULTIPROC_METHOD=spawn
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import tempfile
import copy
# import multiprocessing as mp
import json
VLLM_ENABLE_LENGTH=160000

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
            data.update({'relative_dir': p.replace('.json', '')})
            lines.append(data)
    return lines



def eval_dataset(args):
    min_pixels, max_pixels, context_length = args.min_pixels, args.max_pixels, args.context_length
    model_path = os.path.expanduser(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    if args.rope_scaling == 'ntk':
        config.rope_scaling['rope_type'] = 'ntk'
        config.rope_scaling['factor'] = 4.0
    elif args.rope_scaling == 'default':
        pass
    elif args.rope_scaling == 'dynamic':
        config.rope_scaling['rope_type'] = 'dynamic'
        config.rope_scaling['factor'] = 4.0
        config.rope_scaling["original_max_position_embeddings"] = 32768
    elif args.rope_scaling == 'yarn':
        config.rope_scaling['rope_type'] = 'yarn'
        config.rope_scaling['factor'] = 4.0
        config.rope_scaling["original_max_position_embeddings"] = 32768
    elif args.rope_scaling == 'videoropepp_dynamic':
        config.rope_scaling['rope_type'] = 'videoropepp_dynamic'
        config.rope_scaling['factor'] = 4.0
        config.rope_scaling["original_max_position_embeddings"] = 32768
    elif args.rope_scaling == 'mrope++':
        pass
    elif args.rope_scaling == 'videorope++_v1':
        pass
    elif args.rope_scaling == 'videorope++_v2':
        pass
    else:
        raise ValueError(f"not implemented type {args.rope_scaling}")
    # import pdb; pdb.set_trace()
    if context_length >= VLLM_ENABLE_LENGTH:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        llm = LLM(model_path,
                max_model_len=context_length+1536,
                limit_mm_per_prompt={"video": 10},
                tensor_parallel_size=2,
                gpu_memory_utilization=0.85
                )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, 
                                                            device_map="cpu",
                                                            torch_dtype=torch.bfloat16, 
                                                            attn_implementation="flash_attention_2",
                                                            config=config,
                                                            )
        model = model.to('cuda')
        model = model.eval()
    # import pdb; pdb.set_trace()
    if context_length >= VLLM_ENABLE_LENGTH:
        total_pixels = (context_length - 512) * 28 * 28
    else:
        total_pixels = (context_length - 512) * 28 * 28

    if args.nframes is None:
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, total_pixels=total_pixels)
    else:
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=min_pixels, nframes=args.nframes)
    qa_json = args.Eval_QA_root
    

    data = read_json_file(qa_json)
    if args.clean_subtitles:
        lines = []
        for item in data:
            if 'subtitle' in item['question']:
                continue
            lines.append(item)
        data = copy.deepcopy(lines)
    keys = get_chunk(data, args.num_chunks, args.chunk_idx)
    answer_prompt = "\nAnswer with the option's letter from the given choices directly."

    eval_dict = {}
    
    eval_dataset_json = os.path.join(args.chat_conversation_output_folder, f"{str(args.chunk_idx)}.json")
    os.makedirs(os.path.dirname(eval_dataset_json), exist_ok=True)
    if os.path.exists(eval_dataset_json):
        st = set([item['video_id']+item['question']+str(item['depth']) for item in read_jsonl(eval_dataset_json)])
    else:
        st = set()

    for v_id, _ in tqdm(enumerate(keys)):
        # 清理显存碎片
        torch.cuda.empty_cache()

        item = keys[v_id]
        if item['video_id']+item['question']+str(item['depth']) in st: continue
        def qa_template(data):
            question = f"Question: {data['question']}\n"
            question += "Options:\n"
            answer = data['candidates'][data['correct_choice']]
            answer_idx = -1
            for idx, c in enumerate(data['candidates']):
                question += f"({chr(ord('A') + idx)}) {c}\n"
                if c == answer:
                    answer_idx = idx
            question = question.rstrip()
            answer = f"({chr(ord('A') + answer_idx)}) {answer}"
            return question, answer
        question, answer = qa_template(item)
        # change_prompt
        # question += answer_prompt
        video_path = os.path.join(args.Eval_Video_root, item['video_path'])
        
        total_pixels = (args.context_length-args.max_training_length) * 28 * 28
        noise_text = 'This is noise text, please ignore it.'
        noise_time = args.max_training_length // len(processor.tokenizer.encode(noise_text))
        
        if args.depth == 3.14:
            args.depth = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        else:
            if type(args.depth) is not list:
                args.depth = [args.depth]
        for depth in args.depth:
        # import pdb; pdb.set_trace()
            if args.nframes is None:
                import math

                # 预计算两段噪声文本
                pre_noise  = noise_text * math.floor(noise_time * (1 - depth))
                post_noise = noise_text * math.floor(noise_time *  depth)

                content = []

                # 如果前噪声不为空，加入
                if pre_noise:
                    content.append({"type": "text", "text": pre_noise})

                # 一定要加入的视频项
                content.append({
                    "type": "video",
                    "video": video_path,
                    "min_pixels": min_pixels,
                    # "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                    # "nframes": args.nframes,
                })

                # 如果后噪声不为空，加入
                if post_noise:
                    content.append({"type": "text", "text": post_noise})

                # 最后的提问一定加入
                content.append({"type": "text", "text": question})

                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
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
            # import pdb; pdb.set_trace()
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            # import pdb; pdb.set_trace()
            if context_length < VLLM_ENABLE_LENGTH:
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                inputs = inputs.to(model.device)
                
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, which_rope=args.which_rope, scale_factor=args.rope_scaling, exp_factor=args.exp_factor)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    generated_text = output_text[0]
                print(generated_text)
            else:
                # import pdb; pdb.set_trace()
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                
                print(video_inputs[0].shape[0] // 2 * video_inputs[0].shape[2] // 28 * video_inputs[0].shape[3] // 28)
                # mm_data['which_rope'] = args.which_rope
                # mm_data['scale_factor'] = args.scale_factor
                llm_inputs = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                }
                with torch.no_grad():
                    outputs = llm.generate([llm_inputs], sampling_params=sampling_params, which_rope=args.which_rope, scale_factor=args.scale_factor, scaling_method=args.rope_scaling, exp_factor=args.exp_factor)
                generated_text = outputs[0].outputs[0].text
                print(generated_text)
                del mm_data, llm_inputs, outputs
            # ......
            # =======================================================================================

            pred = copy.deepcopy(_)
            pred.update({'prediction': generated_text, 'answer': answer, 'max_training_length': args.max_training_length, 'context_length': args.context_length, 'depth': depth, 'rope_scaling': args.rope_scaling, 'exp_factor': args.exp_factor})
            save_jsonl_line(eval_dataset_json, pred)

            # 删除无用变量并清理显存
            del messages, prompt, image_inputs, video_inputs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # mp.set_start_method('spawn') useless
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/fs-computility/mllm/weixilin/cache/Qwen2.5-VL-7B-Instruct-t_scale2_change_freq-128frames-16card_8k-context-330k-llava-video")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--nframes", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--Eval_QA_root", type=str,
                        default='/fs-computility/mllm/weixilin/videorope_rebuttal/playground/data/longvideobench/json/lvb_val.json', help="folder containing QA JSON files")
    parser.add_argument("--Eval_Video_root", type=str,
                        default='/fs-computility/mllm/weixilin/videorope_rebuttal/playground/data/longvideobench/videos/', help="folder containing video data")
    parser.add_argument("--chat_conversation_output_folder",
                        type=str, default='/fs-computility/mllm/weixilin/videorope_rebuttal/playground/results/longvideobench/debug', help="")
    
    parser.add_argument("--max_training_length", type=float, default=256000-32768)
    parser.add_argument("--context_length", type=float, default=256000) ## 可以支持156k推理，需要大约max_training_length
    parser.add_argument("--rope_scaling", type=str, default='videorope++_v1')
    parser.add_argument("--exp_factor", type=float, default=4.0)
    parser.add_argument("--depth", type=float, default=0.0)

    parser.add_argument("--min_pixels", type=float, default=144 * 28 * 28)
    parser.add_argument("--max_pixels", type=float, default=256 * 28 * 28)
    parser.add_argument("--which_rope", type=str, default='t_scale2_change_freq')
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--clean_subtitles", action="store_true")
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
