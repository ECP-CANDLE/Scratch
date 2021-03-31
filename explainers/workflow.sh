#!/bin/bash
set -eu

if (( $# != 1 ))
then
  echo "Provide SITE!"
  exit 1
fi

SITE=$1

export THIS=$( readlink --canonicalize $( dirname $0 ) )
source $THIS/env-$SITE.sh

# swift-t $MACHINE workflow.swift $*
# swift-t $MACHINE one-shot.swift $*

stc -u one-shot.swift
turbine $MACHINE -n 4 -e PYTHONPATH=$PYTHONPATH one-shot.tic
