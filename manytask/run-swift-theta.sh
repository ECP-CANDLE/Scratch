#!/bin/bash
set -eu

# RUN SWIFT THETA

date "+%Y-%m-%d %H:%M"

# PATH=/home/wozniak/sfw/theta/swift-t-mt/stc/bin:$PATH
# PATH=/projects/Candle_ECP/swift/2018-06-05/stc/bin:$PATH
# PATH=/projects/Candle_ECP/swift/2018-06-14/stc/bin:$PATH

# SWIFT=/projects/Candle_ECP/swift/2018-06-20
# SWIFT=/projects/Candle_ECP/swift/static
# SWIFT=/projects/Candle_ECP/swift/intel
# SWIFT=/projects/Candle_ECP/swift/2018-08-07
SWIFT=/projects/Swift-T/public/sfw/compute/swift-t/2018-12-10

PATH=$SWIFT/stc/bin:$PATH
PATH=$SWIFT/turbine/bin:$PATH

which swift-t turbine
echo

module swap PrgEnv-intel PrgEnv-gnu
# module load gcc # Loaded by PrgEnv-gnu
# module list
# echo $LD_LIBRARY_PATH | tr ':' '\n'

# export PROJECT=ecp-testbed-01 # Expired 2019-02
# export PROJECT=Candle_ECP # Expired 2019-02
export PROJECT=CSC249ADOA01
# export QUEUE=${QUEUE:-debug-flat-quad}
export WALLTIME=00:30:00

# export TURBINE_APP_RETRIES=3
# export TURBINE_LOG=1

PROGRAM_SWIFT=$1
shift

PROGRAM_TIC=${PROGRAM_SWIFT%.swift}.tic

export THIS=$( readlink --canonicalize $( dirname $0 ) )

SETTINGS=(
  -e PMI_NO_FORK=1
  -e PMI_NO_PREINITIALIZE=1
  -e MPICH_GNI_FORK_MODE=FULLCOPY
)

stc -u $PROGRAM_SWIFT

set -x
turbine -m theta \
        -n $PROCS \
        -w \
        -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
        -e TURBINE_STDOUT="f-%%r.txt" \
        -e THIS \
        ${SETTINGS[@]} \
        $PROGRAM_TIC $*

#         -e PMI_MMAP_SYNC_WAIT_TIME=600


# -e TURBINE_APP_RETRIES_REPUT=3
#         -e TURBINE_LOG_RANKS=1
