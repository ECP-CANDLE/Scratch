#!/bin/bash

echo TASK.SH rank=$ADLB_RANK_SELF $*

ls -ltrh /lustre/atlas/world-shared/stf007/ForArvind/arvind4_ppc64.img
CONTAINER=( /lustre/atlas/world-shared/stf007/ForArvind/arvind4_ppc64.img )

#singularity exec  /lustre/atlas/world-shared/stf007/ForArvind/arvind4_ppc64.img ls /
export SINGULARITY_BINDPATH="$HOME:$HOME"
singularity exec  ${CONTAINER[@]} bash $HOME/singularity_tests/subsing.sh
