#!/bin/bash
# ==================================================
# GLASS — baseline run configuration (ResNet18)
# ==================================================

CUDA_VISIBLE_DEVICES="" python main.py \
  dataset --subdatasets bottle mvtec ../../datasets/mvtec ../../datasets/mvtec \
  net \
  --step 1 \
  --p 0 \
  --lr 0.001 \
  --meta_epochs 1 \
  --eval_epochs 1 \
  --backbone_names resnet18 \
  --layers_to_extract_from layer2 \
  --patchsize 3 \
  --target_embed_dimension 256
