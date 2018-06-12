
import io;
import sys;

n = argv("n");
N = string2int(n);

printf("workers: %i", turbine_workers());
printf("n: %i", n);

app bash(int i)
{
  "bash" "-c" "exit" ; // ("echo "+i) ;
  // "hostname" ;
}

foreach i in [0:N-1]
{
  bash(i);
}
