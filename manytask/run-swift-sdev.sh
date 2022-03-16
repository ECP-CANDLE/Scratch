#!/bin/zsh
set -eu

# SWIFT=/lustre/atlas2/csc249/world-shared/sfw/sdev/swift-t-2018-07-27
# SWIFT=/lustre/atlas2/med106/world-shared/sfw/sdev/swift-t-2018-10-10
SWIFT=/lustre/atlas2/med106/world-shared/sfw/sdev/swift-t-2018-12-27
PATH=$SWIFT/stc/bin:$PATH

which swift-t

source /sw/summit/lmod/7.7.10/rhel7.3_gnu4.8.5/lmod/7.7.10/init/zsh
module load spectrum-mpi

export THIS=$( readlink --canonicalize $( dirname $0 ) )

export PROJECT=MED106
export WALLTIME=14
# echo QUEUE=$QUEUE

checkvars PROCS

PROGRAM_SWIFT=$1
shift
PROGRAM_TIC=${PROGRAM_SWIFT%.swift}.tic
set -x
swift-t -u -o $PROGRAM_TIC \
        -m lsf \
        -n $PROCS \
        -e TURBINE_STDOUT="f-%%r.txt" \
        -e THIS \
        $PROGRAM_SWIFT $*

set -x
sleep 300
brunning -t 120
