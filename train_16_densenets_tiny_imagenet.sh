#!/bin/bash

python train.py --config configs/tiny_imagenet_image_classification/imgcl_densenet_tiny_imagenet_empty.json &&

for i in {0..14}; do
   python train.py --config configs/tiny_imagenet_image_classification/imgcl_densenet_tiny_imagenet_$i.json
done
