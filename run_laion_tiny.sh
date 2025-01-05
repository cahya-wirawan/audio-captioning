#!/bin/sh

ngpu="1" 
dataset_name="cahya/audiosnippets-tiny"
# dataset_name="mitermix/audiosnippets"

# NVidia 4090: disable NCCL_P2P and NCCL_IB
nvidia-smi --query-gpu=name --format=csv,noheader|grep 4090 > /dev/null
if [ $? -eq 0 ]; then
    export NCCL_P2P_DISABLE="1"
    export NCCL_IB_DISABLE="1"
fi

torchrun --nproc_per_node=${ngpu} \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="./checkpoints" \
    --dataset-name="$dataset_name" \
    --train-split=0.99 \
    --max_rows=0 \
    --training-config="./configs/finetune_tiny_config_laion.yaml" \
    --load-checkpoint="../models/whisper-tiny-audio-captioning-v1.5" \
    --wandb-group="finetuning"
