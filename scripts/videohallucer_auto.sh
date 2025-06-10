#!/bin/bash
set -x

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="log/videohallucer_$timestamp.log"
> "$LOG_FILE"

exec > "$LOG_FILE" 2>&1

export CKPT_DIR=/fs-computility/mllm/weixilin/cache
export DECORD_EOF_RETRY_MAX=20480
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list='0,1,2,3,4,5,6,7'
IFS=',' read -ra GPULIST <<< "$gpu_list"
export CHUNKS=${#GPULIST[@]}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export MIN_PIXELS_FACTOR=144


# CONTEXT_LENGTHS=(8192 32768 64000)
# CONTEXT_LENGTHS=(8192 16384 32768 64000)
# CONTEXT_LENGTHS=(32768)
CONTEXT_LENGTHS=(8192 16384 4096 32768)
# CONTEXT_LENGTHS=(32768)
# /mnt/petrelfs/weixilin/cache/Qwen2-VL-t_scale2_change_freq-256frames-16card_8k-context-860k-llava-video-tune_projector-debug

MODEL_LIST=(
    "Qwen2-VL-7B-Instruct-m_rope-128frames-16card_8k-context-330k-llava-video m_rope 1.0"
    "Qwen2-VL-7B-Instruct-t_scale2_change_freq-128frames-16card_8k-context-330k-llava-video t_scale2_change_freq 2.0"
    "Qwen2-VL-7B-Instruct-vanilla_rope-128frames-16card_8k-context-330k-llava-video vanilla_rope 1.0"
        "Qwen2-VL-7B-Instruct-time_rope-128frames-16card_8k-context-330k-llava-video time_rope 1.0"
    "Qwen2-VL-7B-Instruct-ttxy_rope-128frames-16card_8k-context-330k-llava-video  ttxy_rope 2.0"
)

QA_TYPES=("obj_rel" "temporal" "semantic" "fact" "nonfact")


for MODEL_ENTRY in "${MODEL_LIST[@]}"; do

    CKPT=$(echo "$MODEL_ENTRY" | awk '{print $1}')
    WHICH_ROPE=$(echo "$MODEL_ENTRY" | awk '{print $2}')
    SCALE_FACTOR=$(echo "$MODEL_ENTRY" | awk '{print $3}')

    for context_length in "${CONTEXT_LENGTHS[@]}"; do

        export OUTPUT_FOLDER="playground/results/videohallucer/${CKPT}-${context_length}-${MIN_PIXELS_FACTOR}tokens"
        for qa_type in "${QA_TYPES[@]}"; do

            for IDX in $(seq 0 $((CHUNKS-1))); do
                CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m eval.model_videohallucer \
                    --model-path ${CKPT_DIR}/${CKPT} \
                    --max_new_tokens 128 \
                    --Eval_root VideoHallucer \
                    --chat_conversation_output_folder $OUTPUT_FOLDER/${qa_type} \
                    --context_length $context_length \
                    --num-chunks $CHUNKS \
                    --chunk-idx $IDX \
                    --qa_type $qa_type \
                    --which_rope $WHICH_ROPE \
                    --scale_factor $SCALE_FACTOR \
                    --min_pixels $(( MIN_PIXELS_FACTOR * 28 * 28 )) &
            done

            wait
            python eval/check_videohallucer.py $OUTPUT_FOLDER
        done
    done
done
