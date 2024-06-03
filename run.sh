#!/usr/bin/env bash

set -x

code=$GEMINI_CODE/code

input=$GEMINI_DATA_OUT/imagenette_processed_train
output=$GEMINI_DATA_OUT/control-dit_train_baseline-v4
model_type=DiT-XL/2
dit_model=$GEMINI_PRETRAIN2/checkpoints/DiT-XL-2-256x256.pt

image_size=256

epochs=1000
batch_size=128
num_workers=8

python3 $code/train.py --input $input --output $output \
  --model-type $model_type --dit-model-path $dit_model \
  --image-size $image_size --epochs $epochs \
  --batch-size $batch_size --num-workers $num_workers
