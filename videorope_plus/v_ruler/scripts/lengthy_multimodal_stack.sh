#!/bin/bash
set -x

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="log/longvideobench_$timestamp.log"
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

CONTEXT_LENGTHS=(224000 232000 240000 248000 256000 264000 272000 280000 288000 296000)
DEPTH=(3.14)
EXP_FACTOR=(1.0)

ROPE_SCALING=("videorope++_v1")
# ROPE_SCALING=("videorope++_v1" "default" "mrope++" "yarn" "ntk" "dynamic")

MODEL_LIST=(
    # "Qwen2.5-VL-3B-Instruct-m_rope-128frames-16card_8k-context-330k-llava-video m_rope 1.0"
    "Qwen2.5-VL-3B-Instruct-t_scale2_change_freq-128frames-16card_8k-context-330k-llava-video t_scale2_change_freq 2.0"
)
export CHUNKS=$(( ${#GPULIST[@]} / 2 ))

for MODEL_ENTRY in "${MODEL_LIST[@]}"; do

    CKPT=$(echo "$MODEL_ENTRY" | awk '{print $1}')
    WHICH_ROPE=$(echo "$MODEL_ENTRY" | awk '{print $2}')
    SCALE_FACTOR=$(echo "$MODEL_ENTRY" | awk '{print $3}')

    for rope_scaling in "${ROPE_SCALING[@]}"; do
        for context_length in "${CONTEXT_LENGTHS[@]}"; do
            for depth in "${DEPTH[@]}"; do
                for exp_factor in "${EXP_FACTOR[@]}"; do
                    OUTPUT_FOLDER="/fs-computility/mllm/weixilin/videorope_rebuttal/playground/results/longvideobench/v_ruler/6ti_32k_visual_token/${CKPT}/${rope_scaling}-${exp_factor}"
                    max_training_length=$(( context_length - 32768 ))

                    for IDX in $(seq 0 $((CHUNKS-1))); do
                        GPU1=${GPULIST[$((2 * IDX))]}
                        GPU2=${GPULIST[$((2 * IDX + 1))]}
                        VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES="${GPU1},${GPU2}" python -m eval.eval_v_ruler_lengthy_multimodal_stack \
                            --model-path ${CKPT_DIR}/${CKPT} \
                            --max_new_tokens 3 \
                            --Eval_QA_root videorope_plus/v_ruler/data/lengthy_multimodal_stack.json \
                            --chat_conversation_output_folder $OUTPUT_FOLDER/${context_length}/${depth} \
                            --num-chunks $CHUNKS \
                            --chunk-idx $IDX \
                            --which_rope $WHICH_ROPE \
                            --scale_factor $SCALE_FACTOR \
                            --max_training_length $max_training_length \
                            --context_length $context_length \
                            --rope_scaling $rope_scaling \
                            --depth $depth \
                            --exp_factor $exp_factor \
                            --clean_subtitles \
                            --min_pixels $(( MIN_PIXELS_FACTOR * 28 * 28 )) &
                    done
                    wait
                    python eval/check_lengthy_multimodal_stack.py $OUTPUT_FOLDER/${context_length}/${depth}
                done
            done
            wait
            python eval/check_v_ruler_context_level.py $OUTPUT_FOLDER
        done
        python -u eval/plot_v_ruler.py $OUTPUT_FOLDER
    done
done