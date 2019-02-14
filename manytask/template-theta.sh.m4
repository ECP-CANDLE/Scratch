#!/bin/bash
set -eu

define(`getenv_nospace', `esyscmd(printf -- "$`$1'")')dnl

echo "TEMPLATE-THETA START"
echo "DATE: getenv_nospace(DATE)"

module load alps

# which aprun

set -x

PROCS=getenv_nospace(PROCS)
PPN=getenv_nospace(PPN)
TASKS=getenv_nospace(TASKS)

aprun --pes $PROCS -N $PPN ./manytask.x -n $TASKS

echo "TEMPLATE-THETA DONE"

# Local Variables:
# mode: m4
# End:
