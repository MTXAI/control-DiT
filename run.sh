#!/usr/bin/env bash

set -x

out_dir=$1
image_size=$2
epochs=$3
batch_size=$4
num_workers=$5
model_name=$6
logs=$7
ckpts=$8
lr=$9
decay=${10}
model_ckpt=${11}
ema=${12}

code=$GEMINI_CODE/code

input=$GEMINI_DATA_IN1/imagenette_processed_train
output=$GEMINI_DATA_OUT/$out_dir
model_type=DiT-XL/2
dit_model=$GEMINI_PRETRAIN/checkpoints/${model_name}


time accelerate launch $code/train.py --input $input --output $output \
  --model-type $model_type --dit-model-path $dit_model \
  --image-size $image_size --epochs $epochs --batch-size $batch_size \
  --num-workers $num_workers --log-every $logs --ckpt-every $ckpts \
  --lr $lr --decay $decay --model-ckpt $model_ckpt $ema
