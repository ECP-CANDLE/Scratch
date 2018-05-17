#!/bin/bash
set -e

PATH=$HOME/Public/sfw/x86_64/mpich-3.2/bin:$PATH
module load gcc

which mpicc

make CC=$( which mpicc )
