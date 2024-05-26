#!/usr/bin/env bash

set -x
input=$GEMINI_DATA_IN1/imagenette2/train/
output=$GEMINI_DATA_OUT/imagenette_processed_train
code=$GEMINI_CODE/code
index=$code/annotations/imagenet_class_index.json
pretrain=$GEMINI_PRETRAIN/intel-isl_MiDaS_master

python3 $code/preprocess.py --input $input --output $output \
  --num-workers 4 --image-size 256 --class-index $index \
  --deep-model $pretrain --deep-model-source local --batch-size 4
