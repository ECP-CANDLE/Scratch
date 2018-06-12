
#include <assert.h>
#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int
main()
{
  int rc;
  sleep(1);

  pid_t self = getpid();

  rc = kill(self, SIGHUP);
  assert(rc == 0);

  sleep(1);

  printf("How am I alive?\n");
  return 0;
}
