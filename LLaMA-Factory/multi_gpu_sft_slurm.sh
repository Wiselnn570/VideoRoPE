#!/bin/bash
set -x -e
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# export NCCL_IB_HCA="mlx5_0"
# export CUDA_LAUNCH_BLOCKING=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"
export NNODES=2
export num_gpus=8
export WANDB_DISABLED=true
export full_batch_size=128
export batch_size=1
export gradient_accumulation_steps=$[$full_batch_size/($batch_size*$num_gpus*$NNODES)]
export CPUS_PER_TASK=20
export MASTER_PORT=$((RANDOM % 101 + 29400))
export DECORD_EOF_RETRY_MAX=20480
## slurm
export PARTITION=mllm
export JOB_NAME=videorope
export QUOTA_TYPE=reserved
export output_dir=cache/Qwen2-VL-${JOB_NAME}-context-330k-llava-video
export model_name_or_path=cache/Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone

# exec > /mnt/hwfile/mllm/weixilin/log/log_file-16gpu.txt 2>&1
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${num_gpus} \
    --time=14-00:00:00 \
    --quota=$QUOTA_TYPE \
    --nodes=${NNODES} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    bash -c 'torchrun \
    --nnodes $NNODES \
    --nproc_per_node ${num_gpus:-1} \
    --node_rank="${SLURM_NODEID}" \
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \
    --master_port=$MASTER_PORT \
    LLaMA-Factory/src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --model_name_or_path $model_name_or_path \
    --stage sft \
    --total_pixels 6272000 \
    --video_maxlen 128 \
    --do_train true \
    --finetuning_type full \
    --dataset llava_videos_330k_split0,llava_videos_330k_split1,llava_videos_330k_split2 \
    --template qwen2_vl \
    --cutoff_len 8200 \
    --overwrite_cache true \
    --tokenized_path cache/training_qwen2vl_pretokenized_data-llava_videos_330k_other_times300k_2_3mins30k-128frames \
    --preprocessing_num_workers 128 \
    --output_dir $output_dir \
    --num_train_epochs 1.0 \
    --logging_steps 1 \
    --save_steps 800 \
    --save_total_limit 1 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1.0e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --val_size 1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 500 \
    --flash_attn fa2 \
    --which_rope ${JOB_NAME} \
    --report_to none'
