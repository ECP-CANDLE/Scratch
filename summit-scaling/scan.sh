#!/bin/zsh
set -eu

if (( ${#*} != 1 ))
then
  print "Requires directory!"
  return 1
fi

DIR=$1

OUTPUTS=( $DIR/f-*.txt )
if (( ${#OUTPUTS} == 0 ))
then
  echo "No outputs found in $DIR"
  return 1
fi

tclsh scan.tcl $DIR $OUTPUTS
