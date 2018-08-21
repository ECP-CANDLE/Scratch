#!/bin/bash -l

module load torque

set -eu
export NODES=$1

export THIS=$( readlink --canonicalize-existing $( dirname $0 ) )

mkdir -pv run
export OUTPUT=$( mktemp -d run/XXX )
mkdir -pv $OUTPUT
OUTPUT=$( readlink --canonicalize-existing $OUTPUT )
cd $OUTPUT
m4 $THIS/qsub-swan-template.sh > qsub-swan.sh

qsub -l nodes=$NODES \
     -o $OUTPUT/output.txt \
     qsub-swan.sh
