
# ENV SUMMIT COMMON
# Sourced by env-summit-i and env-summit-j

# Let modules initialize LD_LIBRARY_PATH before changing it:
set +eu # modules create errors outside our control
module load spectrum-mpi/10.3.1.2-20200121
module unload darshan-runtime
# module load ibm-wml-ce/1.6.2-3
module list
module load cuda/10.2.89

set -eu

# Swift/T settings
MED106=/gpfs/alpine/world-shared/med106
SWIFT=$MED106/wozniak/sw/gcc-6.4.0/swift-t/2020-10-22
PATH=$SWIFT/stc/bin:$PATH
PATH=$SWIFT/turbine/bin:$PATH

# Python settings
PY=$MED106/sw2/opence010env
LD_LIBRARY_PATH+=:$PY/lib
LD_LIBRARY_PATH+=:/lib64
export PYTHONHOME=$PY
PATH=$PY/bin:$PATH
export PYTHONPATH=$THIS

export LD_LIBRARY_PATH="/sw/summit/gcc/7.4.0/lib64:$LD_LIBRARY_PATH:/sw/summit/gcc/6.4.0/lib64"
