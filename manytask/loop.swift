
import sys;

n = argv("n");
N = string2int(n);

app bash(int i)
{
  "bash" "-c" ("echo "+i) ;
}

foreach i in [0:N-1]
{
  bash(i);
}
