#!/bin/bash
set -e

module swap PrgEnv-intel PrgEnv-gnu
module load gcc
# module list

which cc gcc

export CRAYPE_LINK_TYPE=dynamic
make CC=$( which cc )
