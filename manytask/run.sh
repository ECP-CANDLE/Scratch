# DATE: 2018/05/02 14:25:37
#!bin/bash
set -eu


echo "TEMPLATE-THETA START"

aprun --pes 4096 ./manytask.x 16000

echo "TEMPLATE-THETA DONE"
