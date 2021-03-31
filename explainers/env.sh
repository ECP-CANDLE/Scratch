
# Let modules initialize LD_LIBRARY_PATH before changing it:
set +eu # modules create errors outside our control
module load spectrum-mpi/10.3.1.2-20200121
module unload darshan-runtime
# module load ibm-wml-ce/1.6.2-3
module list
set -eu

# From Wozniak
MED106=/gpfs/alpine/world-shared/med106
# SWIFT=$MED106/sw/gcc-7.4.0/swift-t/2019-10-18  # Python (ibm-wml), no R
# SWIFT=$MED106/sw/gcc-7.4.0/swift-t/2019-11-06  # Python (ibm-wml) and R
# Python (ibm-wml-ce/1.7.0-1) and R:
# SWIFT=$MED106/wozniak/sw/gcc-6.4.0/swift-t/2020-03-31-c
# Python (ibm-wml-ce/1.6.2-3) and R:
# SWIFT=$MED106/wozniak/sw/gcc-6.4.0/swift-t/2020-04-02
# Python (med106/sw/condaenv-200408) and R:
# SWIFT=$MED106/wozniak/sw/gcc-6.4.0/swift-t/2020-04-08
# SWIFT=$MED106/wozniak/sw/gcc-6.4.0/swift-t/2020-04-11
# SWIFT=$MED106/wozniak/sw/gcc-6.4.0/swift-t/2020-08-19
SWIFT=$MED106/wozniak/sw/gcc-6.4.0/swift-t/2020-09-02

export TURBINE_HOME=$SWIFT/turbine
PATH=$SWIFT/stc/bin:$PATH
PATH=$SWIFT/turbine/bin:$PATH
