#!/bin/bash -l

define(`getenv', `esyscmd(printf -- "$`$1'")')

#PBS -l walltime=00:05:00
#PBS -j oe

# #PBS -q getenv(QUEUE)

echo QSUB SWAN START

THIS=getenv(THIS)
NODES=getenv(NODES)
OUTPUT=getenv(OUTPUT)

TCLSH=/home/users/p02473/Public/sfw/tcl-8.6.8/bin/tclsh8.6

export LD_LIBRARY_PATH=$THIS

echo OUTPUT=$OUTPUT
cd $OUTPUT

aprun -n 4 -N 1 $TCLSH $THIS/td.tcl

echo QSUB SWAN STOP

# Local Variables:
# mode: m4;
# End:
