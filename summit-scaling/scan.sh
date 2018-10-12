#!/bin/zsh
set -eu

DIR=$1

set -x

OUTPUTS=( $DIR/f-*.txt )
if (( ${#OUTPUTS} == 0 ))
then
  echo "No outputs found in $DIR"
  return 1
fi

tclsh scan.tcl $DIR $OUTPUTS
