#!/bin/bash -l

module load torque

set -eu
export NODES=$1
export PPN=$2

export PROCS=$(( NODES * PPN ))
export TASKS=1 # $(( PROCS * 2 ))

export THIS=$( readlink --canonicalize-existing $( dirname $0 ) )

mkdir -pv run
export OUTPUT=$( mktemp -d run/XXX )
mkdir -pv $OUTPUT
OUTPUT=$( readlink --canonicalize-existing $OUTPUT )
cd $OUTPUT
m4 $THIS/qsub-swan-template.sh > qsub-swan.sh

set -x
qsub qsub-swan.sh
