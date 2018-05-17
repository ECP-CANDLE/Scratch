#!/bin/bash
set -e

module swap PrgEnv-intel PrgEnv-gnu
module load gcc
module list

which cc

make CC=$( which cc )
