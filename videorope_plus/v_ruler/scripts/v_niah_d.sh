#!/bin/bash
set -x

models=(
  "/fs-computility/mllm/shared/weixilin/share_model/Qwen2.5-VL-7B-Instruct-m_rope-128frames-16card_8k-context-330k-llava-video"
  "/fs-computility/mllm/shared/weixilin/share_model/Qwen2.5-VL-7B-Instruct-t_scale2_change_freq-128frames-16card_8k-context-330k-llava-video"
  "/fs-computility/mllm/weixilin/cache/Qwen2.5-VL-7B-Instruct-vanilla_rope-128frames-16card_8k-context-330k-llava-video"
  "/fs-computility/mllm/weixilin/cache/Qwen2.5-VL-7B-Instruct-time_rope-128frames-16card_8k-context-330k-llava-video"
)

# 对应的 rope_type
rope_types=(
  "m_rope"
  "t_scale2_change_freq"
  "vanilla_rope"
  "time_rope"
)

# basic port
base_port=6011

# iterate each model
for i in "${!models[@]}"; do
  model=${models[$i]}
  rope_type=${rope_types[$i]}
  
  port=$((base_port + i))

  echo "evaluating model: $model"
  echo "using rope_type: $rope_type"
  echo "port: $port"

  accelerate launch --num_processes 8 --config_file easy_context/accelerate_configs/deepspeed_inference.yaml \
    --main_process_port "$port" eval/eval_v_ruler_interrupt.py \
    --model "$model" \
    --needle_dataset videorope_plus/v_ruler/data/v_niah_d.json \
    --embedding_by_name_dir /fs-computility/mllm/weixilin/videorope_rebuttal/vision_niah/outputs/needle_embeddings_by_name \
    --haystack_dir data/haystack_vicuna_embeddings_6000frames_qwen2.5 \
    --prompt_template qwen2 \
    --max_frame_num 3000 \
    --min_frame_num 100 \
    --frame_interval 200 \
    --output_path /fs-computility/mllm/weixilin/videorope_rebuttal/vision_niah/outputs/v_ruler_v_niah_d_6_3-5_ti \
    --rope_type "$rope_type" \
    --image_tokens 144 \
    --depth_interval 0.2

  echo "model $model evaluation has done."
  echo "------------------------------------"
done
