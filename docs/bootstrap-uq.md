1. 	modified:   workflows/common/sh/langs-app-cori.sh
  
    +export KMP_BLOCKTIME=1
    +export KMP_SETTINGS=1
    +export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
    +export OMP_NUM_THREADS=68
    +export NUM_INTER_THREADS=1
    +export NUM_INTRA_THREADS=68

2. 	modified:   workflows/common/sh/model.sh

    +# to monitor performance. Remove for production runs
    +USER="$( whoami )"
    +HOST="$( hostname )"
    +function_to_fork() {
    +  top -b -n 60 -d 60 -u $USER > $COBALT_JOBID.$HOST.$$.top
    +}
    +function_to_fork &

3. 	modified:   workflows/upf/swift/workflow.sh
  
    +export TURBINE_LAUNCH_OPTIONS="-cc none"

4. 	modified:   workflows/upf/test/cfg-sys-1.sh
  
    -export PROCS=${PROCS:-4}
    +export PROCS=${PROCS:-10}
    
5. 	modified:   workflows/upf/test/upf-1.sh
  
    -export MODEL_NAME="p1b1"
    +export MODEL_NAME="combo"

6. 	modified:   workflows/upf/test/upf-1.txt
