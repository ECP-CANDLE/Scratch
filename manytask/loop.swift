
import io;
import sys;

n = argv("n");
N = string2int(n);

this = getenv("THIS");
task = this/"task.sh";

printf("workers: %i", turbine_workers());
printf("tasks:   %i", n);

printf("PMI_NO_FORK:          '%s'", getenv("PMI_NO_FORK"));
printf("PMI_NO_PREINITIALIZE: '%s'", getenv("PMI_NO_PREINITIALIZE"));
printf("MPICH_GNI_FORK_MODE:  '%s'", getenv("MPICH_GNI_FORK_MODE"));

app bash(string task, int i)
{
  "bash" task ;
    // "-c" "exit" ; // ("echo "+i) ;
  // "bash" "-c" "exit" ; // ("echo "+i) ;
  // "bash" "/home/wozniak/proj/Scratch/manytask/task.sh" i ;
  // "hostname" ;
  // "/bin/bash" "/home/wozniak/proj/SV/workflows/common/sh/model.sh"  ; // i ; // "keras" "NONE" i
  // "/home/wozniak/proj/Scratch/manytask/task.x" ;
}

foreach i in [0:N-1]
{
  bash(task, i);
}
