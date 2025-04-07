#!/bin/bash

python converted_CIFAR100.py --device 2 --VthHand 1.0 --sin_t 256 &

python converted_CIFAR100.py --device 5 --VthHand 1.0 --sin_t 16 &

python converted_CIFAR100.py --device 8 --VthHand 1.0 --sin_t 32 &