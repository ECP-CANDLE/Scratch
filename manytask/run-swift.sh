#!/bin/bash

PATH=/home/wozniak/sfw/theta/swift-t-mt/stc/bin:$PATH

which swift-t

module swap PrgEnv-intel PrgEnv-gnu
# module list
# exit

export TURBINE_LOG=1
set -x
swift-t -m theta -t w -n $PROCS -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH $*
