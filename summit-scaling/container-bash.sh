#!/bin/bash

o TASK.SH rank=$ADLB_RANK_SELF $*

ls -ltrh /lustre/atlas/world-shared/stf007/ForArvind/arvind4_ppc64.img
CONTAINER=( /lustre/atlas/world-shared/stf007/ForArvind/arvind4_ppc64.img )
export SINGULARITY_BINDPATH="$HOME:$HOME"

singularity exec ${CONTAINER[@]} bash /ccs/home/jain/Scratch/summit-scaling/test_bash.sh
