 python sample.py --dit-ckpt ./pretrained_models/DiT-XL-2-256x256.pt \
        --control-dit-ckpt ./pretrained_models/ControlDiT-XL-2-256x256.pt \
        --deep-model ./pretrained_models/intel-isl_MiDaS_master \
        --vae-model ./pretrained_models/sd-vae-ft-ema \
        --test-image ./datasets/COCO_Captions/example2017/tarinval/000000190236.jpg \
        --deep-model-source local

