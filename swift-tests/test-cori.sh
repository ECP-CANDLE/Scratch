#!/bin/zsh

set -eu

# pt /projects/Candle_ECP/swift/2018-06-05/stc/bin
# pt ~/sfw/theta/swift-t-mt/stc/bin
# pt /projects/Candle_ECP/swift/2018-06-20/stc/bin
# SWIFT=/projects/Candle_ECP/swift/intel
# SWIFT=/projects/Candle_ECP/swift/2018-08-07
# SWIFT=$HOME/Public/sfw/theta/swift-t/2018-08-22b
# SWIFT=/projects/Candle_ECP/swift/2018-03-07
#SWIFT=$HOME/Public/sfw/theta/swift-t/2018-08-22b
SWIFT=/global/homes/w/wozniak/Public/sfw/compute/swift-t-2018-06-05
PATH=$SWIFT/stc/bin:$SWIFT/turbine/bin:$PATH

export PROJECT=debug #ecp-testbed-01
export QUEUE=m2924 #debug-cache-quad
# export QUEUE=debug-flat-quad
export WALLTIME=00:02:00
export PPN=2

export TURBINE_DIRECTIVE="#SBATCH -C knl,quad,cache\n#SBATCH --license=SCRATCH"


#PYTHONHOME="/lus/theta-fs0/projects/Candle_ECP/ncollier/py2_tf_gcc6.3_eigen3_native"
#R=/projects/Candle_ECP/swift/deps/R-3.4.1/lib64/R

PYTHONHOME=/global/common/cori/software/python/2.7-anaconda/envs/deeplearning

R=/global/homes/w/wozniak/Public/sfw/R-3.4.0-gcc-7.1.0/lib64/R

LLP=$R/lib

# WORKFLOW=hi.swift
WORKFLOW=pyr.swift

swift-t -V -m slurm\
        -p \
        -t w \
        -e LD_LIBRARY_PATH=$LLP \
        -e PYTHONHOME=$PYTHONHOME \
        /global/project/projectdirs/m2924/rjain/Scratch/swift-tests/$WORKFLOW

# turbine -m theta \
#         -w \
#         hi.tic


#         -e PYTHONHOME=$PYTHONHOME \

# ~/mcs/ste/py-keras.swift

# ~/mcs/ste/py0.swift

        #

#        ~/mcs/ste/app-py.swift

        # ~/mcs/ste/app2.swift
        # -e TURBINE_LOG=1 \
        # -e TURBINE_APP_DELAY=10 \

date_nice
