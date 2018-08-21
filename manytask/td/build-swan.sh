#!/bin/bash -l

module load PrgEnv-gnu

export CRAYPE_LINK_TYPE=dynamic

export CC=cc

make
