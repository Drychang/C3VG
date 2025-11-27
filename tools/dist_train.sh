#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

PORT=${PORT:-29511}

PYTHON=/home/drychang/miniconda3/envs/c3vg_py38/bin/python

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
$PYTHON -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
