#!/bin/zsh

cd $1/; cat jobid.txt; cat $1/f-* |grep Time:;cat $1/turbine.log |grep NODES; cat $1/turbine.log |grep PROCS;cat $1/turbine.log |grep PPN; grep -ri er $1/f-002.txt |sort -u; grep -ri er $1/f-02.txt|sort -u |grep -ri er $1/f-2.txt |sort -u
