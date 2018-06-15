#!/bin/bash
set -eu

# PATH=/home/wozniak/sfw/theta/swift-t-mt/stc/bin:$PATH
# PATH=/projects/Candle_ECP/swift/2018-06-05/stc/bin:$PATH
PATH=/projects/Candle_ECP/swift/2018-06-14/stc/bin:$PATH

which swift-t

module swap PrgEnv-intel PrgEnv-gnu
# module list
# exit

export PROJECT=ecp-testbed-01
export QUEUE=${QUEUE:-debug-flat-quad}

# export TURBINE_APP_RETRIES=3
export TURBINE_LOG=1

set -x
swift-t -m theta \
        -n $PROCS \
        -t w \
        -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
        -e TURBINE_APP_RETRIES_REPUT=3 \
        -e TURBINE_LOG_RANKS=1 \
        $*
