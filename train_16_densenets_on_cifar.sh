#!/bin/bash

echo python train.py --config configs/cifar_image_classification/imgcl_densenet_cifar_3.json &&

for i in {1..14}; do
   echo python train.py --config configs/cifar_image_classification/imgcl_densenet_cifar_$i.json
done
