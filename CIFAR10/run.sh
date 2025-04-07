#!/bin/bash

python converted_CIFAR10.py --device 0 --VthHand 0.5 &
PID1=$!;

python converted_CIFAR10.py --device 1 --VthHand 0.5 --model_name resnet20&
PID2=$!;
wait ${PID1} && wait ${PID2}

python converted_CIFAR10.py --device 0 --VthHand 0.7 &
PID1=$!;

python converted_CIFAR10.py --device 1 --VthHand 0.7 --model_name resnet20&
PID2=$!;
wait ${PID1} && wait ${PID2}

python converted_CIFAR10.py --device 0 --VthHand 0.9 &
PID1=$!;

python converted_CIFAR10.py --device 1 --VthHand 0.9 --model_name resnet20&
PID2=$!;
wait ${PID1} && wait ${PID2}

python converted_CIFAR10.py --device 0 --VthHand -1 --useDET&
PID1=$!;

python converted_CIFAR10.py --device 1 --VthHand -1 --useDET --model_name resnet20&
PID2=$!;
wait ${PID1} && wait ${PID2}

python converted_CIFAR10.py --device 0 --VthHand -1 --useDTT&
PID1=$!;

python converted_CIFAR10.py --device 1 --VthHand -1 --useDTT --model_name resnet20&
PID2=$!;
wait ${PID1} && wait ${PID2}

python converted_CIFAR10.py --device 0 --VthHand -1 --useDET --useDTT&
PID1=$!;

python converted_CIFAR10.py --device 1 --VthHand -1 --useDET --useDTT --model_name resnet20&
PID2=$!;
wait ${PID1} && wait ${PID2}

python converted_CIFAR10.py --device 0 --VthHand -1 --useDET --useDTT --useSC&
PID1=$!;

python converted_CIFAR10.py --device 1 --VthHand -1 --useDET --useDTT --useSC --model_name resnet20&
PID2=$!;
wait ${PID1} && wait ${PID2}
