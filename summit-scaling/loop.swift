
import io;
import sys;

n = argv("n");
N = string2int(n);
task = argv("task");

this = getenv("THIS");

printf("workers: %i", turbine_workers());
printf("tasks:   %i", n);

app bash(string this, string task, int i)
{
  "bash" (this/task) ;
    // "-c" "exit" ; // ("echo "+i) ;
  // "hostname" ;
}

foreach i in [0:N-1]
{
  bash(this, task, i);
}
