
# ENV SUMMIT JOB

source $THIS/env-summit-common.sh

MACHINE="-m lsf"

# Default PROJECT for CANDLE
#export QUEUE=${QUEUE:-batch-hm}
export PROJECT=${PROJECT:-MED106}

export PPN=4
