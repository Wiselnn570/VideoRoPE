#!/bin/bash
set -x

models=(
  # "/mnt/petrelfs/weixilin/cache/Qwen2-VL-t_scale2_change_freq-128frames-16card_8k-context-330k-llava-video"
  # "/mnt/petrelfs/weixilin/cache/Qwen2-VL-m_rope-128frames-16card_8k-context-330k-llava-video"
  "/mnt/petrelfs/weixilin/cache/Qwen2-VL-vanilla_rope-128frames-16card_8k-context-330k-llava-video"
  "/mnt/petrelfs/weixilin/cache/Qwen2-VL-time_rope-128frames-16card_8k-context-330k-llava-video"
  # "/mnt/petrelfs/weixilin/cache/Qwen2-VL-t_only-128frames-16card_8k-context-330k-llava-video"
)
rope_types=(
  # "t_scale2_change_freq" 
  "vanilla_rope"
  "time_rope"
  # "t_only"
  )

base_port=6015

for i in "${!models[@]}"; do
  model=${models[$i]}
  rope_type=${rope_types[$i]}
  
  port=$((base_port + i))

  echo "evaluating model: $model"
  echo "using rope_type: $rope_type"
  echo "port: $port"

  accelerate launch --num_processes 8 --config_file vision_niah_d/easy_context/accelerate_configs/deepspeed_inference.yaml \
    --main_process_port "$port" vision_niah_d/eval_vision_niah.py \
    --model "$model" \
    --needle_dataset vision_niah_d/needle_datasets/dataset.json \
    --needle_embedding_dir vision_niah_d/video_needle_haystack/data/needle_qwen2_embeddings_144tokens_dataset \
    --haystack_dir vision_niah_d/video_needle_haystack/data/haystack_qwen2_embeddings_6000frames \
    --prompt_template qwen2 \
    --max_frame_num 3000 \
    --min_frame_num 100 \
    --frame_interval 200 \
    --output_path vision_niah_d/niah_output/ \
    --rope_type "$rope_type" \
    --image_tokens 144 \
    --depth_interval 0.2

  echo "model $model evaluation has done."
  echo "------------------------------------"
done
