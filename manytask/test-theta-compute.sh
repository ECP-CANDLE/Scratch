#!/bin/bash
set -eu

if (( ${#} != 2 ))
then
  echo "Requires node count, task count!"
  exit 1
fi

export PROCS=$1
export TASKS=$2
WALLTIME=${WALLTIME:-00:05:00}
# WALLTIME=00:30:00

export PPN=${PPN:-1}
NODES=$(( PROCS / PPN ))

echo PROCS=$PROCS PPN=$PPN NODES=$NODES TASKS=$TASKS WALLTIME=$WALLTIME

# Export to m4
export DATE=$( date "+%Y-%m-%d_%H:%M:%S" )
OUTPUT=out-$DATE.txt
echo OUTPUT=$OUTPUT
echo

{
  m4 < template-theta.sh.m4
} > run.sh
chmod u+x run.sh

export QUEUE=${QUEUE:-default}
# QUEUE=debug-cache-quad

MAIL_ARG="--notify woz@anl.gov"

JOB=$( qsub --project   CSC249ADOA01 \
            --queue     $QUEUE \
            --nodecount $NODES \
            --time      $WALLTIME \
            --output    $OUTPUT \
            --error     $OUTPUT \
            $MAIL_ARG \
            run.sh )

echo JOB=$JOB | tee -a $OUTPUT

cqwait $JOB

echo JOB COMPLETE $( date "+%Y/%m/%d %H:%M" )
