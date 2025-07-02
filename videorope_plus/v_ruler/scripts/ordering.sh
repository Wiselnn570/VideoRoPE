#!/bin/bash
set -x

# 模型名称和对应的 image_tokens
models=(
  "/fs-computility/mllm/weixilin/cache/Qwen2.5-VL-7B-Instruct-t_scale2_change_freq-128frames-16card_8k-context-330k-llava-video"
  "/fs-computility/mllm/weixilin/cache/Qwen2.5-VL-7B-Instruct-m_rope-128frames-16card_8k-context-330k-llava-video"
  "/fs-computility/mllm/weixilin/cache/Qwen2.5-VL-7B-Instruct-vanilla_rope-128frames-16card_8k-context-330k-llava-video"
  "/fs-computility/mllm/weixilin/cache/Qwen2.5-VL-7B-Instruct-time_rope-128frames-16card_8k-context-330k-llava-video"
) # 添加更多模型路径
rope_types=(
  "t_scale2_change_freq"
  "m_rope"
  "vanilla_rope"
  "time_rope"
)

# 基础端口号（避免使用占用的端口范围，如 6000-7000）
base_port=6015

# 遍历每个模型和对应的 image_tokens
for i in "${!models[@]}"; do
  model=${models[$i]}
  rope_type=${rope_types[$i]}
  
  # 动态计算端口号（基础端口号 + 索引值，确保每次唯一）
  port=$((base_port + i))

  echo "正在评测模型: $model" 
  echo "使用的 rope_type: $rope_type"
  echo "分配的端口号: $port"

  # 运行评测命令
  accelerate launch --num_processes 8 --config_file easy_context/accelerate_configs/deepspeed_inference.yaml \
    --main_process_port "$port" eval/eval_v_ruler_order_frames.py \
    --model "$model" \
    --needle_dataset videorope_plus/v_ruler/data/ordering.json \
    --needle_embedding_dir vision_niah/outputs/needle_embeddings_by_name \
    --haystack_dir data/haystack_vicuna_embeddings_6000frames_qwen2.5 \
    --prompt_template qwen2 \
    --max_frame_num 3000 \
    --min_frame_num 100 \
    --frame_interval 200 \
    --output_path /fs-computility/mllm/weixilin/videorope_rebuttal/vision_niah/outputs/v_ruler_order_frames_6_5-5_ti \
    --rope_type "$rope_type" \
    --image_tokens 144 \
    --depth_interval 0.2

  echo "模型 $model 的评测完成"
  echo "------------------------------------"
done
