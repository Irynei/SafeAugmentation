#!/bin/bash

python train.py --config configs/imgcl_densenet_cifar_empty.json &&
python train.py --config configs/imgcl_densenet_cifar_0.json &&
python train.py --config configs/imgcl_densenet_cifar_1.json &&
python train.py --config configs/imgcl_densenet_cifar_2.json &&
python train.py --config configs/imgcl_densenet_cifar_3.json &&
python train.py --config configs/imgcl_densenet_cifar_4.json &&
python train.py --config configs/imgcl_densenet_cifar_5.json &&
python train.py --config configs/imgcl_densenet_cifar_6.json
