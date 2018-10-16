#!/bin/bash

TASK.SH rank=$ADLB_RANK_SELF $*

ls -ltrh /lustre/atlas/world-shared/stf007/ForArvind/arvind4_ppc64.img
CONTAINER=( /lustre/atlas/world-shared/stf007/ForArvind/arvind4_ppc64.img )
export SINGULARITY_BINDPATH="$HOME:$HOME"
singularity exec --nv ${CONTAINER[@]}  python3 /ccs/home/jain/Scratch/summit-scaling/keras_test.py
