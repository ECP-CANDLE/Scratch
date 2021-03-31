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

# swift-t workflow.swift $*
# swift-t one-shot.swift $*

stc -u one-shot.swift
turbine -n 4 -e PYTHONPATH=$PYTHONPATH one-shot.tic
