#!/usr/bin/env bash

set -x

image_size=$1
epochs=$2
batch_size=$3
num_workers=$4
model_name=$5

code=$GEMINI_CODE/code

input=$GEMINI_DATA_IN1/imagenette_processed_train
output=$GEMINI_DATA_OUT/control-dit_train_baseline-v4
model_type=DiT-XL/2
dit_model=$GEMINI_PRETRAIN/checkpoints/${model_name}
vae_model=$GEMINI_PRETRAIN2/sd-vae-ft-ema


time accelerate launch $code/train.py --input $input --output $output \
  --model-type $model_type --dit-model-path $dit_model \
  --vae-model-path $vae_model --image-size $image_size \
  --epochs $epochs --batch-size $batch_size --num-workers $num_workers
