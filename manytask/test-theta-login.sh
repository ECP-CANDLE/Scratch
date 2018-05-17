#!/bin/bash
set -eu

if (( ${#} != 2 ))
then
  echo "Requires node count, task count!"
  exit 1
fi

NODES=$1
TASKS=$2

DATE=$( date "+%Y-%m-%d_%H:%M:%S" )
OUTPUT=out-$DATE.txt

PATH=$HOME/Public/sfw/x86_64/mpich-3.2/bin:$PATH
module load gcc

which mpiexec

set -x
# {
  nice mpiexec -n $NODES ./manytask.x $TASKS
# } > $OUTPUT
