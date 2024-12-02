#!/bin/bash -l
set -eu

SWIFT=/lustre/atlas2/csc249/world-shared/sfw/sdev/swift-t-2018-07-27
PATH=$SWIFT/stc/bin:$PATH

which swift-t

module load spectrum-mpi

export THIS=$( readlink --canonicalize $( dirname $0 ) )

#export PROJECT=CSC249ADOA01
export PROJECT=MED106
export WALLTIME=30
# echo QUEUE=$QUEUE

swift-t -m lsf \
        -n $PROCS \
        -e TURBINE_STDOUT="f-%%r.txt" \
	-e THIS \
        $*

sleep 60
# brunning -t 60
