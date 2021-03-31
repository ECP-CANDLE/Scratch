#!/bin/bash
set -eu

if (( $# != 2 ))
then
  echo "Provide SITE and WORKFLOW!"
  exit 1
fi

SITE=$1
WORKFLOW=$2

export THIS=$( readlink --canonicalize $( dirname $0 ) )
source $THIS/env-$SITE.sh

stc -u $WORKFLOW.swift
turbine $MACHINE -n 4 -e PYTHONPATH=$PYTHONPATH $WORKFLOW.tic
