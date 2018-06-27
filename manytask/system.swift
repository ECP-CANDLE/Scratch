
import io;
import sys;

n = argv("n");
N = string2int(n);

printf("workers: %i", turbine_workers());
printf("N count: %i", N);

foreach i in [0:N-1]
{
  system1("/bin/bash /home/wozniak/proj/SV/workflows/common/sh/model.sh");
}
