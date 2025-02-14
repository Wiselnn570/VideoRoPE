import argparse
import gc
import sys
import torch
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from easy_context import Qwen2ForCausalLM_RingAttn
from tqdm import tqdm
from accelerate import Accelerator
import glob
import numpy as np
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from pathlib import Path
import random
import json
from datasets import load_dataset
from vision_niah.produce_needle_embedding import read_json_file
from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
)
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
apply_seq_parallel_monkey_patch("zigzag_ring_attn", "llama")

import sys
import pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

SEED = 24242424
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
IMAGE_TOKENS = None
prompt_templates = {
    "mistral": {
        "preprompt": "<s>[INST]",
        "postprompt": " [/INST]"
    },
    "vicuna": {
        "preprompt": "<s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:",
        "postprompt": "ASSISTANT:"
    },
    "llama3": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "qwen2": {
        "preprompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    }, 
    "yi": {
        "preprompt": "<|im_start|>system\nAnswer the questions.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    },
}
# \nAnswer the question using a single word or phrase.
# The color of the bottle cap is
# answer = "Yellow"


def safe_tokenize(tokenizer, text):
    tokenized = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.bos_token != None and len(tokenized) > 0 and tokenized[0, 0] == tokenizer.bos_token_id:
        tokenized = tokenized[:, 1:]
    return tokenized

def get_vanilla_rope_index(input_embeds, video_se):
    return torch.arange(input_embeds.shape[1]).view(1, 1, -1).expand(3, 1, -1)

def get_time_rope_index(input_embeds, video_se):
    llm_pos_ids_list = []
    llm_pos_ids_list.append(torch.arange(video_se[0]).view(1, 1, -1).expand(3, 1, -1))
    assert (video_se[1] - video_se[0]) % IMAGE_TOKENS == 0, 'frames should not be float'
    nframes = (video_se[1] - video_se[0]) // IMAGE_TOKENS
    ## time rope
    t_index = torch.arange(llm_pos_ids_list[-1].max().item() + 1, llm_pos_ids_list[-1].max().item() + 1 + nframes).repeat_interleave(IMAGE_TOKENS, dim=0).view(1, 1, -1).expand(3, 1, -1)
    llm_pos_ids_list.append(t_index)
    if input_embeds.shape[1] > video_se[1]:
        text_len = input_embeds.shape[1] - video_se[1]
        llm_pos_ids_list.append(torch.arange(t_index.max().item() + 1, text_len + t_index.max().item() + 1).view(1, 1, -1).expand(3, 1, -1))
    position_ids = torch.cat(llm_pos_ids_list, dim=-1)
    assert position_ids.shape[-1] == input_embeds.shape[1], f'shape mismatch! {position_ids.shape[-1]=}, {input_embeds.shape[1]=}'
    return position_ids

def get_t_scale2_rope_index(input_embeds, video_se, scale_factor):
    llm_pos_ids_list = []
    llm_pos_ids_list.append(torch.arange(video_se[0]).view(1, 1, -1).expand(3, 1, -1))
    st_idx = llm_pos_ids_list[-1][0].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    assert (video_se[1] - video_se[0]) % IMAGE_TOKENS == 0, 'frames should not be float'
    nframes = (video_se[1] - video_se[0]) // IMAGE_TOKENS
    ## m_rope rope
    llm_grid_t, llm_grid_h, llm_grid_w = nframes, 9, 16
    
    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
        -1, llm_grid_h * llm_grid_w).flatten()
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
        llm_grid_t, -1, llm_grid_w).flatten() - (llm_grid_h-1) // 2
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
        llm_grid_t, llm_grid_h, -1).flatten() - (llm_grid_w-1) // 2
    t_index = t_index * scale_factor
    t_index = t_index + st_idx
    h_index = h_index + t_index
    w_index = w_index + t_index
    
    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]).unsqueeze(dim=1))
    
    if input_embeds.shape[1] > video_se[1]:
        text_len = input_embeds.shape[1] - video_se[1]
        # print(text_len)
        
        llm_pos_ids_list.append(torch.arange(llm_pos_ids_list[-1][0].max().item() + 1, llm_pos_ids_list[-1][0].max().item() + 1 + text_len).view(1, 1, -1).expand(3, 1, -1))
        # print(llm_pos_ids_list[0].shape, llm_pos_ids_list[1].shape, llm_pos_ids_list[2].shape)
    position_ids = torch.cat(llm_pos_ids_list, dim=-1)
    assert position_ids.shape[-1] == input_embeds.shape[1], f'shape mismatch! {position_ids.shape[-1]=}, {input_embeds.shape[1]=}'
    return position_ids

def get_m_rope_index(input_embeds, video_se):
    llm_pos_ids_list = []
    llm_pos_ids_list.append(torch.arange(video_se[0]).view(1, 1, -1).expand(3, 1, -1))
    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    assert (video_se[1] - video_se[0]) % IMAGE_TOKENS == 0, 'frames should not be float'
    nframes = (video_se[1] - video_se[0]) // IMAGE_TOKENS
    ## m_rope rope
    llm_grid_t, llm_grid_h, llm_grid_w = nframes, 9, 16
    t_index = torch.arange(nframes).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]).unsqueeze(dim=1) + st_idx)
    if input_embeds.shape[1] > video_se[1]:
        text_len = input_embeds.shape[1] - video_se[1]
        llm_pos_ids_list.append(torch.arange(llm_pos_ids_list[-1].max().item() + 1, llm_pos_ids_list[-1].max().item() + 1 + text_len).view(1, 1, -1).expand(3, 1, -1))
    position_ids = torch.cat(llm_pos_ids_list, dim=-1)
    assert position_ids.shape[-1] == input_embeds.shape[1], f'shape mismatch! {position_ids.shape[-1]=}, {input_embeds.shape[1]=}'
    return position_ids

def get_m_modify_margin_index(input_embeds, video_se):
    llm_pos_ids_list = []
    llm_pos_ids_list.append(torch.arange(video_se[0]).view(1, 1, -1).expand(3, 1, -1))
    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    assert (video_se[1] - video_se[0]) % IMAGE_TOKENS == 0, 'frames should not be float'
    nframes = (video_se[1] - video_se[0]) // IMAGE_TOKENS
    ## m_rope rope
    llm_grid_t, llm_grid_h, llm_grid_w = nframes, 9, 16
    
    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
        -1, llm_grid_h * llm_grid_w).flatten()
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
        llm_grid_t, -1, llm_grid_w).flatten() - (llm_grid_h-1) // 2
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
        llm_grid_t, llm_grid_h, -1).flatten() - (llm_grid_w-1) // 2
    t_index = t_index + st_idx
    h_index = h_index + t_index
    w_index = w_index + t_index
    
    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]).unsqueeze(dim=1))
    
    if input_embeds.shape[1] > video_se[1]:
        text_len = input_embeds.shape[1] - video_se[1]
        # print(text_len)
        
        llm_pos_ids_list.append(torch.arange(llm_pos_ids_list[-1].max().item() + 1, llm_pos_ids_list[-1].max().item() + 1 + text_len).view(1, 1, -1).expand(3, 1, -1))
        # print(llm_pos_ids_list[0].shape, llm_pos_ids_list[1].shape, llm_pos_ids_list[2].shape)
    position_ids = torch.cat(llm_pos_ids_list, dim=-1)
    assert position_ids.shape[-1] == input_embeds.shape[1], f'shape mismatch! {position_ids.shape[-1]=}, {input_embeds.shape[1]=}'
    return position_ids

def get_m_modify_no_center_index(input_embeds, video_se):
    llm_pos_ids_list = []
    llm_pos_ids_list.append(torch.arange(video_se[0]).view(1, 1, -1).expand(3, 1, -1))
    st_idx = llm_pos_ids_list[-1][0].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    assert (video_se[1] - video_se[0]) % IMAGE_TOKENS == 0, 'frames should not be float'
    nframes = (video_se[1] - video_se[0]) // IMAGE_TOKENS
    ## m_rope rope
    llm_grid_t, llm_grid_h, llm_grid_w = nframes, 9, 16
    
    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
        -1, llm_grid_h * llm_grid_w).flatten()
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
        llm_grid_t, -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
        llm_grid_t, llm_grid_h, -1).flatten()
    
    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]).unsqueeze(dim=1) + st_idx)
    
    if input_embeds.shape[1] > video_se[1]:
        text_len = input_embeds.shape[1] - video_se[1]
        # print(text_len)
        
        llm_pos_ids_list.append(torch.arange(llm_pos_ids_list[-1][0].max().item() + 1, llm_pos_ids_list[-1][0].max().item() + 1 + text_len).view(1, 1, -1).expand(3, 1, -1))
        # print(llm_pos_ids_list[0].shape, llm_pos_ids_list[1].shape, llm_pos_ids_list[2].shape)
    position_ids = torch.cat(llm_pos_ids_list, dim=-1)
    assert position_ids.shape[-1] == input_embeds.shape[1], f'shape mismatch! {position_ids.shape[-1]=}, {input_embeds.shape[1]=}'
    return position_ids

def get_m_modify_index(input_embeds, video_se):
    llm_pos_ids_list = []
    llm_pos_ids_list.append(torch.arange(video_se[0]).view(1, 1, -1).expand(3, 1, -1))
    st_idx = llm_pos_ids_list[-1][0].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    assert (video_se[1] - video_se[0]) % IMAGE_TOKENS == 0, 'frames should not be float'
    nframes = (video_se[1] - video_se[0]) // IMAGE_TOKENS
    ## m_rope rope
    llm_grid_t, llm_grid_h, llm_grid_w = nframes, 9, 16
    
    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
        -1, llm_grid_h * llm_grid_w).flatten()
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
        llm_grid_t, -1, llm_grid_w).flatten() - (llm_grid_h-1) // 2
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
        llm_grid_t, llm_grid_h, -1).flatten() - (llm_grid_w-1) // 2
    t_index = t_index + st_idx
    h_index = h_index + t_index
    w_index = w_index + t_index
    
    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]).unsqueeze(dim=1))
    
    if input_embeds.shape[1] > video_se[1]:
        text_len = input_embeds.shape[1] - video_se[1]
        # print(text_len)
        
        llm_pos_ids_list.append(torch.arange(llm_pos_ids_list[-1][0].max().item() + 1, llm_pos_ids_list[-1][0].max().item() + 1 + text_len).view(1, 1, -1).expand(3, 1, -1))
        # print(llm_pos_ids_list[0].shape, llm_pos_ids_list[1].shape, llm_pos_ids_list[2].shape)
    position_ids = torch.cat(llm_pos_ids_list, dim=-1)
    assert position_ids.shape[-1] == input_embeds.shape[1], f'shape mismatch! {position_ids.shape[-1]=}, {input_embeds.shape[1]=}'
    return position_ids
    
def get_position_ids(input_embeds, rope_type, video_se):
    if rope_type == 'vanilla_rope':
        return get_vanilla_rope_index(input_embeds, video_se)
    elif rope_type == 'tad_rope':
        return get_time_rope_index(input_embeds, video_se) + get_vanilla_rope_index(input_embeds, video_se)
    elif rope_type == 'm_rope':
        return get_m_rope_index(input_embeds, video_se)
    elif rope_type == 'videorope':
        scale_factor = 2.0
        return get_t_scale2_rope_index(input_embeds, video_se, scale_factor)
    else:
        raise ValueError(f"not this rope: {rope_type}")

# answer = "more bet"
def eval_forward(args, video_se, accelerator, model, input_embeds, answer_embeds, pad_id, answer_ids, tokenizer):
    # first append answer_embeds to input_embeds
    prompt_length = input_embeds.shape[1]
    labels_length = answer_embeds.shape[1]
    input_embeds = torch.cat([input_embeds, answer_embeds], dim=1)
    # second pad input_embeds to the multiple of accelerator.num_processes
    pad_tensor = torch.tensor(
        [pad_id]
        * (
            (accelerator.num_processes * 2)
            - input_embeds.shape[1] % (accelerator.num_processes * 2)
        )
    ).unsqueeze(0).unsqueeze(-1).expand(-1, -1, input_embeds.shape[-1]).to(accelerator.device)
    input_embeds = torch.cat([input_embeds, pad_tensor], dim=1)
    # position_ids = (
    #     torch.arange(input_embeds.shape[1]).unsqueeze(0).expand(input_embeds.shape[0], -1)
    # ).to(accelerator.device)
    position_ids = get_position_ids(input_embeds, args.rope_type, video_se)
    # ForkedPdb().set_trace()
    accelerator.print(input_embeds.shape)
    prepared = prepare_seq_parallel_inputs(
        "zigzag_ring_attn",
        input_embeds,
        position_ids,
        None,
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )
    local_input_embeds = prepared["local_input_ids"]
    local_position_ids = prepared["local_position_ids"]
    if 'm_modify' in args.rope_type or 't_only' in args.rope_type or 'change_freq' in args.rope_type:
        from transformers.models.qwen2_vl import modeling_qwen2_vl
        modeling_qwen2_vl.apply_multimodal_rotary_pos_emb = modeling_qwen2_vl.apply_m_modify_multimodal_rotary_pos_emb
    with torch.inference_mode():
        hidden_states = model.model(
            inputs_embeds=local_input_embeds,
            position_ids=local_position_ids,
            use_cache=False,
        )[0]
        logits = model.lm_head(hidden_states)
        logits = logits.float()

        pred = logits.argmax(dim=-1)

    # gather all logits using accelerator.gather
    def undo_extract_local(gathered_value, world_size, dim=1):
        value_chunks = gathered_value.chunk(2 * world_size, dim=dim)
        reordered_chunks = [None] * (2 * world_size)
        for i in range(world_size):
            reordered_chunks[i] = value_chunks[i * 2]
            reordered_chunks[2 * world_size - i - 1] = value_chunks[i * 2 + 1]
        return torch.cat(reordered_chunks, dim=dim)

    correct = False

    gathered_logits = accelerator.gather(pred.squeeze(0)).unsqueeze(0)
    # undo extract local on the gathered logits
    # ForkedPdb().set_trace()
    pred = undo_extract_local(gathered_logits, accelerator.num_processes)
    pred = pred[:, prompt_length - 1 : prompt_length + labels_length - 1]
    # check if the logits are correct, extract argmax id
    # compare the predicted_ids with the labels 
    correct = (pred == answer_ids.to(accelerator.device)).all()
    if  accelerator.is_main_process:
        print(
            "Predicted: ",
            tokenizer.decode(pred.squeeze().tolist()),
            "Answer: ",
            tokenizer.decode(answer_ids.squeeze().tolist()),
        )
        # print id as well
        print(
            "Predicted: ",
            pred.squeeze().tolist(),
            "Answer: ",
            answer_ids.squeeze().tolist(),
        )
    return int(correct)


def load_haystack(args, accelerator):
    haystack_embeddings = torch.load(f"{args.haystack_dir}/video_embeddings.pt").to(torch.bfloat16)

    return haystack_embeddings

def load_text_embeddings(str, tokenizer, model, accelerator, replace_double_newline=False): 
    token_ids = safe_tokenize(tokenizer, str)
    def replace_double_newline_func(token_ids):
        # subsitute token id 271 to two 198]
        # for example:
        # from: tensor([[128000, 128006,   9125, 128007,    271,   2675,    527,    264,  11190, 4221,    323,  11376,  18328,     13]])
        # to: tensor([[128000, 128006,   9125, 128007,    198,    198,    2675,    527,    264,  11190, 4221,    323,  11376,  18328,     13]])
        # length will increase by number of 271
        double_newline_loc = (token_ids == 271).nonzero()[:, 1]
        double_newline_loc += torch.arange(len(double_newline_loc))
        if len(double_newline_loc) > 0:
            for loc in double_newline_loc:
                token_ids = torch.cat([token_ids[:, :loc], torch.tensor([[198, 198]]), token_ids[:, loc+1:]], dim=1)
        return token_ids
    if replace_double_newline:
        token_ids = replace_double_newline_func(token_ids)
    token_ids = token_ids.to(accelerator.device)
    with torch.inference_mode():
        embeddings = model.model.embed_tokens(token_ids)
    return embeddings.to(torch.bfloat16)

def inference(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
    )
    model_path = args.model
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, 
                                                        device_map=accelerator.device,
                                                        torch_dtype=torch.bfloat16, 
                                                        attn_implementation="flash_attention_2"
                                                        )
    del model.visual
    processor = AutoProcessor.from_pretrained("/mnt/hwfile/mllm/weixilin/cache/Qwen2-VL-7B-Instruct")
    tokenizer = processor.tokenizer
    
    
    kwargs = {"rope_theta": args.rope_theta} if args.rope_theta is not None else {}
    tokenizer.pad_token = tokenizer.eos_token
    # remember to remove <s>
    accelerator.print("Preparing Haystack...")
    haystack_embeddings = load_haystack(args, accelerator)
    target_length = args.max_frame_num * IMAGE_TOKENS
    # ForkedPdb().set_trace()
    if len(haystack_embeddings) < target_length:
        repeat_times = (target_length + len(haystack_embeddings) - 1) // len(haystack_embeddings)  # 向上取整计算需要重复的次数
        haystack_embeddings = torch.cat([haystack_embeddings] * repeat_times, dim=0)[:target_length]

    assert len(haystack_embeddings) >= args.max_frame_num * IMAGE_TOKENS, "Haystack embeddings are not enough. Max frame {} is not found. Currently only {} frames.".format(args.max_frame_num, len(haystack_embeddings))
    # import pdb; pdb.set_trace()
    
    haystack_embeddings = haystack_embeddings[:args.max_frame_num * IMAGE_TOKENS].to(accelerator.device)
    prompt = prompt_templates[args.prompt_template]
    preprompt_embeddings = load_text_embeddings(prompt["preprompt"], tokenizer, model, accelerator, args.replace_double_newline)
    postprompt_embeddings = load_text_embeddings(prompt["postprompt"], tokenizer, model, accelerator, args.replace_double_newline)
    
    needle_dataset = read_json_file(args.needle_dataset)
    answer_embedding_list = []
    answer_id_list = []
    needle_embedding_list = []
    needle_embedding_interrupt_list = []
    question_embeding_list = []
    for index, instance in enumerate(needle_dataset):
        answer = instance["answer"]
        question = instance["prompt"]
        needle_embedding_list.append(torch.load(args.needle_embedding_dir + f"/{index}.pt", map_location="cpu").to(torch.bfloat16).to(accelerator.device))
        needle_embedding_interrupt_list.append(torch.load(args.needle_embedding_interrupt_dir + f"/{index}.pt", map_location="cpu").to(torch.bfloat16).to(accelerator.device))
        answer_embedding_list.append(load_text_embeddings(answer, tokenizer, model, accelerator))
        answer_id_list.append(safe_tokenize(tokenizer, answer))
        question_embeding_list.append(load_text_embeddings(question, tokenizer, model, accelerator))
        
    accelerator.print("Starting Evaluation...")
    model = accelerator.prepare(model)
    model.gradient_checkpointing_enable()
    all_accuries = []
    for num_frames in tqdm(
        range(
            args.min_frame_num, args.max_frame_num + 1, args.frame_interval
        )
    ):
        for depth in np.arange(0, 1 + args.depth_interval, args.depth_interval):
            accuracies = []
            for question_embedding, needle_embedding, needle_embedding_interrupt, answer_embedding, answer_id in zip(question_embeding_list, needle_embedding_list, needle_embedding_interrupt_list, answer_embedding_list, answer_id_list):
                query_frame_idx = int(depth * num_frames)
                # import pdb; pdb.set_trace()
                #! interrupt every 200 frames
                import random
                test_p = random.random()
                mode = 'no'
                cycle = 200
                if query_frame_idx - cycle <= 0 and query_frame_idx + cycle >= num_frames: mode = 'no'
                elif query_frame_idx < cycle: mode = 'after'
                elif query_frame_idx + cycle >= num_frames:  mode = 'before'
                elif test_p < 0.5: mode = 'before'
                else: mode  = 'after'
                print(f"{mode=}")
                if mode == 'before':
                    input_frames = torch.cat([haystack_embeddings[:(query_frame_idx-cycle) * IMAGE_TOKENS].to(accelerator.device), needle_embedding_interrupt.to(accelerator.device), haystack_embeddings[(query_frame_idx-cycle) * IMAGE_TOKENS:query_frame_idx * IMAGE_TOKENS].to(accelerator.device), needle_embedding.to(accelerator.device), haystack_embeddings[query_frame_idx*IMAGE_TOKENS:num_frames*IMAGE_TOKENS].to(accelerator.device)], dim=0).view(-1, haystack_embeddings.shape[-1]).unsqueeze(0)
                elif mode == 'after':
                    input_frames = torch.cat([haystack_embeddings[:query_frame_idx * IMAGE_TOKENS].to(accelerator.device),needle_embedding.to(accelerator.device), haystack_embeddings[query_frame_idx*IMAGE_TOKENS:(query_frame_idx+cycle)*IMAGE_TOKENS].to(accelerator.device), needle_embedding_interrupt.to(accelerator.device), haystack_embeddings[(query_frame_idx+cycle)*IMAGE_TOKENS:num_frames*IMAGE_TOKENS].to(accelerator.device)], dim=0).view(-1, haystack_embeddings.shape[-1]).unsqueeze(0)
                else:
                    input_frames = torch.cat([haystack_embeddings[:query_frame_idx * IMAGE_TOKENS].to(accelerator.device),needle_embedding.to(accelerator.device), haystack_embeddings[query_frame_idx*IMAGE_TOKENS:num_frames*IMAGE_TOKENS].to(accelerator.device)], dim=0).view(-1, haystack_embeddings.shape[-1]).unsqueeze(0)
                input_emebds = torch.cat([preprompt_embeddings.to(accelerator.device), input_frames.to(accelerator.device),question_embedding.to(accelerator.device), postprompt_embeddings.to(accelerator.device)], dim=1)
                video_se = (preprompt_embeddings.shape[1], preprompt_embeddings.shape[1] + input_frames.shape[1])
                # ForkedPdb().set_trace()
                correct = eval_forward(
                    args, video_se, accelerator, model, input_emebds, answer_embedding, tokenizer.pad_token_id, answer_id, tokenizer
                )
                gc.collect()
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    accuracies.append(correct)
            if accelerator.is_main_process:
                result = {
                    "Num. Frame": num_frames,
                    "Frame Depth": round(depth * 100, -1),
                    "Score": sum(accuracies) / len(accuracies),
                }
                accelerator.print(result)
                all_accuries.append(result)
    if accelerator.is_main_process:
        model_name = args.model.split("/")[-1]
        os.makedirs(f"{args.output_path}/{model_name}", exist_ok=True)
        # save all_accuries as json
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "w") as f:
            json.dump(all_accuries, f, indent=4)
    return all_accuries, accelerator


def plot(args,  all_accuries):
    df = pd.DataFrame(all_accuries)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#9ad5b3"]
    )

    pivot_table = pd.pivot_table(
        df,
        values="Score",
        index=["Frame Depth", "Num. Frame"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(
        index="Frame Depth", columns="Num. Frame", values="Score"
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
    plt.xlabel("Num. of Frames", fontsize=14)  # X-axis label
    plt.ylabel("Depth Percent", fontsize=14)  # Y-axis label
    plt.xticks(ticks=[i + 0.5 for i in range(len(context_lengths))], labels=formatted_context_lengths, rotation=45, fontsize=14)
    # plt.xticks(rotation=45, fontsize=14)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=14)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    model_name = args.model.split("/")[-1]

    plt.savefig(f"{args.output_path}/{model_name}/heatmap.png")
    # calculate average accuracy
    average_accuracy = df["Score"].mean()
    print(f"Average Accuracy: {average_accuracy}")
    # save as txt
    with open(f"{args.output_path}/{model_name}/avg_accuracy.txt", "w") as f:
        f.write(f"Average Accuracy: {average_accuracy}\n")
        
def main(args):
    if args.plot_only:
        # load all_accuracies from json
        model_name = args.model.split("/")[-1]
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "r") as f:
            all_accuracies = json.load(f)
        plot(args, all_accuracies)
    else:
        all_accuracies, accelerator = inference(args)
        if accelerator.is_main_process:
            plot(args, all_accuracies)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="/mnt/hwfile/mllm/weixilin/cache/Qwen2-VL-7B-Instruct")
    args.add_argument("--max_frame_num", type=int, default=1500)
    args.add_argument("--needle_dataset", type=str, default="/mnt/petrelfs/weixilin/projects/MLLM/Qwen2-VL/vision_niah/needle_datasets/dataset_change_format.json")
    args.add_argument("--min_frame_num", type=int, default=400)
    args.add_argument("--frame_interval", type=int, default=100)
    args.add_argument("--output_path", type=str, default="/mnt/petrelfs/weixilin/projects/MLLM/Qwen2-VL/vision_niah/niah_output")
    args.add_argument("--depth_interval", type=float, default=0.1)
    args.add_argument("--num_samples", type=int, default=1)
    args.add_argument("--rope_theta", type=float, default=None)
    args.add_argument("--haystack_dir", type=str, default="/mnt/petrelfs/weixilin/projects/MLLM/Qwen2-VL/vision_niah/video_needle_haystack/data/haystack_vicuna_embeddings_concat3000frames_144tokens_has_background")
    args.add_argument("--needle_embedding_dir", type=str, default="/mnt/petrelfs/weixilin/projects/MLLM/Qwen2-VL/vision_niah/video_needle_haystack/data/needle_vicuna_embeddings")
    args.add_argument("--needle_embedding_interrupt_dir", type=str, default="/mnt/petrelfs/weixilin/projects/MLLM/Qwen2-VL/vision_niah/video_needle_haystack/data/needle_vicuna_embeddings")
    args.add_argument("--prompt_template", type=str, default='qwen2')
    args.add_argument("--image_tokens", type=int, default=144)
    args.add_argument("--rope_type", type=str, default=None)
    args.add_argument("--replace_double_newline", action="store_true")
    args.add_argument("--plot_only", action="store_true")
    args = args.parse_args()
    IMAGE_TOKENS = args.image_tokens
    main(args)
