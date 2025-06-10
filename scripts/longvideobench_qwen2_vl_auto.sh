#!/bin/bash
set -x

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="log/longvideobench_$timestamp.log"
> "$LOG_FILE"

exec > "$LOG_FILE" 2>&1

export CKPT_DIR=/your/path/to/ckpt/dir
export DECORD_EOF_RETRY_MAX=20480
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list='0,1,2,3,4,5,6,7'
IFS=',' read -ra GPULIST <<< "$gpu_list"
export CHUNKS=${#GPULIST[@]}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export MIN_PIXELS_FACTOR=144

CONTEXT_LENGTHS=(8192 16384 32768 64000)

MODEL_LIST=(
    "Qwen2-VL-m_rope-128frames-16card_8k-context-330k-llava-video m_rope 1.0"
    "Qwen2-VL-videorope-128frames-16card_8k-context-330k-llava-video videorope 2.0"
)

for MODEL_ENTRY in "${MODEL_LIST[@]}"; do
    CKPT=$(echo "$MODEL_ENTRY" | awk '{print $1}')
    WHICH_ROPE=$(echo "$MODEL_ENTRY" | awk '{print $2}')
    SCALE_FACTOR=$(echo "$MODEL_ENTRY" | awk '{print $3}')
    for context_length in "${CONTEXT_LENGTHS[@]}"; do
        OUTPUT_FOLDER="playground/results/longvideobench/${CKPT}-${context_length}-${MIN_PIXELS_FACTOR}tokens-clean_subtitles"
        
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m eval.model_longvideobench_qwen2_vl \
                --model-path ${CKPT_DIR}/${CKPT} \
                --max_new_tokens 128 \
                --Eval_QA_root playground/data/longvideobench/json/lvb_val.json \
                --chat_conversation_output_folder $OUTPUT_FOLDER \
                --context_length $context_length \
                --num-chunks $CHUNKS \
                --chunk-idx $IDX \
                --which_rope $WHICH_ROPE \
                --scale_factor $SCALE_FACTOR \
                --clean_subtitles \
                --min_pixels $(( MIN_PIXELS_FACTOR * 28 * 28 )) &
        done

        wait
        
        python eval/check_long_video_bench.py $OUTPUT_FOLDER
    done
done