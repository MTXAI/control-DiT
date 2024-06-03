accelerate launch train.py --input ./datasets/tiny-imagenet-200_processed_train \
    --output ./output/control-dit_train_baseline-v4 --model-type DiT-B/2 \
    --dit-model-path ./pretrained_models/DiT-XL-2-256x256.pt --image-size 256 \
    --epochs 1000 --batch-size 1 --num-workers 0