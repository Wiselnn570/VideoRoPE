from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
import argparse
import numpy as np
from tqdm import tqdm
import torch
import transformers
import math
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import os
import json
IMAGE_FACTOR = 28
MIN_PIXELS = 144 * 28 * 28
MAX_PIXELS = 144 * 28 * 28
MAX_RATIO = 200

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

def read_json_file(file_path):
    """
    读取JSON文件并返回数据作为字典。

    参数:
    file_path (str): JSON文件的路径。

    返回:
    dict: JSON文件中的数据。
    """
    try:
        # 打开文件并读取数据
        with open(file_path, 'r', encoding='utf-8') as file:
            # 将JSON数据解析为字典
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def fetch_image(ele, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = 252, 448
        # resized_height, resized_width = smart_resize(
        #     ele["resized_height"],
        #     ele["resized_width"],
        #     factor=size_factor,
        # )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        # resized_height, resized_width = smart_resize(
        #     height,
        #     width,
        #     factor=size_factor,
        #     min_pixels=min_pixels,
        #     max_pixels=max_pixels,
        # )
        resized_height, resized_width = 252, 448
    image = image.resize((resized_width, resized_height))
    return image
def main(args):
    model_path = args.model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, 
                                                        device_map="auto",
                                                        torch_dtype=torch.bfloat16, 
                                                        attn_implementation="flash_attention_2"
                                                        )
    processor = AutoProcessor.from_pretrained("/fs-computility/mllm/weixilin/cache/Qwen2.5-VL-7B-Instruct-t_scale2_change_freq-128frames-16card_8k-context-330k-llava-video")
    del model.model.layers
    # dataset = load_dataset(args.needle_dataset)["test"]
    # dataset = load_dataset('json', '/mnt/petrelfs/weixilin/projects/MLLM/LongVA/vision_niah/needle_datasets/dataset.json')
    # dataset = read_json_file(args.needle_dataset)
    image_path_list = os.listdir(args.image_list_path)
    for index, image_path in enumerate(image_path_list):
        # import pdb; pdb.set_trace()
        # image = instance["image"].convert("RGB")
        image_abs_path = os.path.join(args.image_list_path, image_path)
        if '.png' not in image_abs_path: continue
        img = fetch_image({"image": image_abs_path, "resized_height": 252, "resized_width": 448})
        image_single = processor.image_processor(images=[img], videos=None)
        merge_length = processor.image_processor.merge_size**2
        pixel_values, image_grid_thw=torch.from_numpy(image_single['pixel_values']), torch.from_numpy(image_single['image_grid_thw']).to(model.device)
        # import pdb; pdb.set_trace()
        # pixel_values = pixel_values.type(model.visual.get_dtype()).to(model.device)
        pixel_values = pixel_values.to(model.device)
        image_embed = model.visual(pixel_values, grid_thw=image_grid_thw).to(model.device)
        print(image_embed.shape)
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(image_embed, f"{args.output_dir}/{image_path.split('.')[0]}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/fs-computility/mllm/weixilin/cache/Qwen2.5-VL-7B-Instruct-t_scale2_change_freq-128frames-16card_8k-context-330k-llava-video")
    parser.add_argument("--image_list_path", type=str, default="/fs-computility/mllm/weixilin/videorope_rebuttal/vision_niah/needle_datasets/images")
    parser.add_argument("--output_dir", type=str, default="/fs-computility/mllm/weixilin/videorope_rebuttal/vision_niah/outputs/needle_embeddings_by_name")
    args = parser.parse_args()
    main(args)
