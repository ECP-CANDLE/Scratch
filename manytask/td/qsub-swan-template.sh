changecom(`dnl')#!/bin/bash -l

define(`getenv', `esyscmd(printf -- "$`$1'")')

#PBS -l nodes=getenv(NODES):ppn=getenv(PPN)
#PBS -l walltime=00:01:00
#PBS -o getenv(OUTPUT)/output.txt
#PBS -j oe

# #PBS -q getenv(QUEUE)

echo
echo QSUB SWAN START

export THIS=getenv(THIS)
NODES=getenv(NODES)
PPN=getenv(PPN)
PROCS=getenv(PROCS)
OUTPUT=getenv(OUTPUT)
TASKS=getenv(TASKS)

TCLSH=/home/users/p02473/Public/sfw/tcl-8.6.8/bin/tclsh8.6

export LD_LIBRARY_PATH=$THIS

echo THIS=$THIS
echo OUTPUT=$OUTPUT
cd $OUTPUT

ENVS=( -e THIS=$THIS )

echo
(
    set -x
    aprun ${ENVS[@]} \
      -n $PROCS -N $PPN \
      $TCLSH $THIS/td.tcl $TASKS
    CODE=$?
    if (( $CODE ))
    then
        echo FAIL: APRUN CODE: $CODE
    fi
)

echo QSUB SWAN STOP

# Local Variables:
# mode: m4;
# End:
