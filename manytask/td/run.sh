#!/bin/zsh
set -eu

MACHINE=swan

./build-$MACHINE.sh
./run-$MACHINE.sh   $*
sleep 30
qrunning -c -l -t 1 | cat
echo
echo OK
