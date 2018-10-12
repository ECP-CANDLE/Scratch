#!/bin/bash

echo TASK-EXIT.SH rank=$ADLB_RANK_SELF $*

exit 0

# CONTAINER=( /lustre/atlas/world-shared/stf007/ForArvind/arvind4_ppc64.img
#             --bind $HOME:$HOME )

# Interactive:
# singularity shell ${CONTAINER[@]}

# One shot:
# singularity exec ${CONTAINER[@]} bash subtask.sh

# Real fast:
# singularity exec ${CONTAINER[@]} bash -c exit
