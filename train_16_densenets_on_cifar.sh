#!/bin/bash

echo python train.py --config configs/imgcl_densenet_cifar_empty.json &&

for i in {1..14}; do
   echo python train.py --config configs/imgcl_densenet_cifar_$i.json
done
