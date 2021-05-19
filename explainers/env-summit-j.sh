
# ENV SUMMIT JOB

source $THIS/env-summit-common.sh

MACHINE="-m lsf"

# Default PROJECT for CANDLE
#export QUEUE=${QUEUE:-batch-hm}
export PROJECT=${PROJECT:-MED106}

# Adjust as needed:
export WALLTIME=00:42:00
export PROCS=${PROCS:-15}
# MPI processes per node.  This should not exceed PROCS.
export PPN=1
