#!/bin/bash

set -x

# 模型列表
models=(
  "/fs-computility/mllm/shared/weixilin/share_model/Qwen2.5-VL-7B-Instruct-m_rope-128frames-16card_8k-context-330k-llava-video"
  "/fs-computility/mllm/shared/weixilin/share_model/Qwen2.5-VL-7B-Instruct-t_scale2_change_freq-128frames-16card_8k-context-330k-llava-video"
)

# 对应的 rope_type
rope_types=(
  "m_rope"
  "t_scale2_change_freq"
)

task_name="counting"
embeddings_dir_name="counting"

# output_path 可选单独设置；否则自动生成
# output_base="/fs-computility/mllm/weixilin/videorope_rebuttal/vision_niah/outputs/debug_test_multi_key_multi_value"
output_base="outputs/$task_name"

# 端口基数（避免冲突）
base_port=6005

for i in "${!models[@]}"; do
  model=${models[$i]}
  rope_type=${rope_types[$i]}
  port=$((base_port + i))

  # 自动构造 output_path 后缀（也可使用固定名）
  # output_path="${output_base}_${i}"

  echo "运行模型: $model"
  echo "Rope type: $rope_type"
  echo "端口: $port"
  echo "输出路径: $output_base"

  accelerate launch --num_processes 8 --config_file easy_context/accelerate_configs/deepspeed_inference.yaml \
  --main_process_port "$port" eval/eval_v_ruler_counting.py \
  --model "$model" \
  --needle_dataset needle_datasets/dataset_dsy/$task_name.json \
  --needle_embedding_dir embeddings/$embeddings_dir_name \
  --haystack_dir video_needle_haystack/data/haystack_vicuna_embeddings_6000frames_qwen2.5 \
  --prompt_template qwen2 \
  --max_frame_num 3000 \
  --min_frame_num 100 \
  --frame_interval 200 \
  --output_path "$output_base" \
  --rope_type "$rope_type" \
  --image_tokens 144 \
  --depth_interval 0.2

  echo "完成模型: $model"
  echo "------------------------------------"
done
