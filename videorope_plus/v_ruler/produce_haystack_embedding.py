from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import transformers
import math
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
IMAGE_FACTOR = 28
MIN_PIXELS = 144 * 28 * 28
MAX_PIXELS = 144 * 28 * 28
MAX_RATIO = 200
def load_video_batches(video_path, batch_size):
    global args
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    for start_idx in range(0, len(frame_idx), batch_size):
        end_idx = min(start_idx + batch_size, total_frame_num)
        frame_indices = frame_idx[start_idx:end_idx]
        batch_frames = vr.get_batch(frame_indices).asnumpy()
        batch_frames = torch.tensor(batch_frames).permute(0, 3, 1, 2)
        # import pdb; pdb.set_trace()
        nframes, _, height, width = batch_frames.shape
        # if torch.unique(batch_frames).numel() == 1:
        #     batch_frames.fill_(args.v)
        #     print(torch.unique(batch_frames).item())
        resized_height, resized_width = 252, 448
        # resized_height, resized_width = smart_resize(
        #         height,
        #         width,
        #         factor=IMAGE_FACTOR,
        #         min_pixels=MIN_PIXELS,
        #         max_pixels=MAX_PIXELS,
        #     )
        batch_frames = transforms.functional.resize(
            batch_frames,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        yield batch_frames

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor
def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def main(args):
    video_path = args.video_path
    model_path = args.model
    model_name = "llava_qwen"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, 
                                                        device_map="auto",
                                                        torch_dtype=torch.bfloat16, 
                                                        attn_implementation="flash_attention_2"
                                                        )
    processor = AutoProcessor.from_pretrained(model_path)
    del model.model.layers
    # Process video in batches 
    batch_size = 32
    total_batches = (args.sampled_frames_num + batch_size - 1) // batch_size
    image_feature_list = []
    if args.add_newline_token:
        newline_token_embeddong = model.model.image_newline
    with torch.inference_mode():
        for i, video_batch in tqdm(enumerate(load_video_batches(video_path, batch_size)), total=total_batches, desc="Processing Video Batches"):
            v_test = processor.image_processor(images=None, videos=video_batch)
            merge_length = processor.image_processor.merge_size**2
            pixel_values_videos,video_grid_thw=torch.from_numpy(v_test['pixel_values_videos']), torch.from_numpy(v_test['video_grid_thw']).to(model.device)
            # if i > 30:
            #     import pdb; pdb.set_trace()
            print(video_grid_thw)
            # import pdb; pdb.set_trace()
            # pixel_values_videos = pixel_values_videos.type(model.visual.get_dtype()).to(model.device)
            pixel_values_videos = pixel_values_videos.to(model.device)
            video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw).to(model.device)
            
            print(video_embeds.shape)
            if args.add_newline_token:
                image_features = torch.cat([image_features, newline_token_embeddong.unsqueeze(0).expand(image_features.shape[0], 1, -1)], dim=1)
            image_feature_list.append(video_embeds.to(torch.bfloat16).to("cpu"))
            if i > total_batches:
                break
    image_feature_list = torch.cat(image_feature_list, dim=0)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(image_feature_list, f"{args.output_dir}/video_embeddings.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/fs-computility/mllm/weixilin/cache/Qwen2.5-VL-7B-Instruct-m_rope-128frames-16card_8k-context-330k-llava-video")
    # parser.add_argument("--v", type=int, default=255)
    parser.add_argument("--video_path", type=str, default="/fs-computility/mllm/weixilin/videorope_rebuttal/asset/movie.mp4")
    parser.add_argument("--sampled_frames_num", type=int, default=6000)
    parser.add_argument("--output_dir", type=str, default="/fs-computility/mllm/weixilin/videorope_rebuttal/vision_niah/video_needle_haystack/data/haystack_vicuna_embeddings_6000frames")
    parser.add_argument("--pooling_size", type=int, default=0)
    parser.add_argument("--add_newline_token", action="store_true")
    args = parser.parse_args()
    main(args)
