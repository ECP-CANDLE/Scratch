#!bin/bash
set -eu

define(`getenv_nospace', `esyscmd(printf -- "$`$1'")')dnl

echo "TEMPLATE-THETA START"

aprun --pes getenv_nospace(NODES) ./manytask.x getenv_nospace(TASKS)

echo "TEMPLATE-THETA DONE"
