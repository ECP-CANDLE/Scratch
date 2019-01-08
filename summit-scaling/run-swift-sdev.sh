#!/bin/bash
set -eu

. /sw/summit/lmod/7.7.10/rhel7.3_gnu4.8.5/lmod/7.7.10/init/bash

# SWIFT=/lustre/atlas2/csc249/world-shared/sfw/sdev/swift-t-2018-07-27
SWIFT=/lustre/atlas2/med106/world-shared/sfw/sdev/swift-t-2018-10-10
PATH=$SWIFT/stc/bin:$PATH

which swift-t

module load spectrum-mpi

export THIS=$( readlink --canonicalize $( dirname $0 ) )

export PROJECT=MED106 # CSC249ADOA01
export WALLTIME=30
# echo QUEUE=$QUEUE

set -x
swift-t -m lsf \
        -n $PROCS \
        -e TURBINE_STDOUT="f-%%r.txt" \
        -e THIS \
        $*

# sleep 60
# brunning -t 60
