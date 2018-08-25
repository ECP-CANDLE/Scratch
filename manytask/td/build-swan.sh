#!/bin/bash -l

module unload PrgEnv-gnu
module load PrgEnv-gnu

export CRAYPE_LINK_TYPE=dynamic

export CC=cc
which cc

export TCL=NOPE

set -x
make
