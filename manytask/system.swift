
import io;
import sys;

n = argv("n");
N = string2int(n);

printf("workers: %i", turbine_workers());
printf("N count: %i", N);

foreach i in [0:N-1]
{
  (output,rc) =
    system1("/bin/bash /home/wozniak/proj/Scratch/manytask/task.sh " + i);
  if (rc == 0)
  {
    printf(output);
  }
  else
  {
    printf("task failed: %i", i);
  }
}
