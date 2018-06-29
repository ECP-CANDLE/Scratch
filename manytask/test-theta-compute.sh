#!/bin/bash
set -eu

if (( ${#} != 2 ))
then
  echo "Requires node count, task count!"
  exit 1
fi

export NODES=$1
export TASKS=$2
WALLTIME=${WALLTIME:-00:30:00}

DATE=$( date "+%Y-%m-%d_%H:%M:%S" )
OUTPUT=out-$DATE.txt

{
  echo "# DATE: $DATE"
  m4 < template-theta.sh.m4
} > run.sh
chmod u+x run.sh

# debug-cache-quad
QUEUE=default
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

echo JOB=$JOB

cqwait $JOB
