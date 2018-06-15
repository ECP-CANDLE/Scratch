
#include <assert.h>
#include <errno.h>
#include <sys/types.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static double rand_double(void);
static void crash(char* fmt, ...);

/**
    This program accepts:
    1) a chance to kill itself with SIGHUP
    2) some characters to echo to stdout
    The self-kill tests Swift/T's task reput feature.
    The characters are useful to test the string processing in Swift/T's
    task reput feature.  Quotes, curly braces, etc. could cause problems.
*/
int
main(int argc, char* argv[])
{
  if (argc != 3)
    crash("requires 2 arguments: <chance to self-kill> <characters>");

  pid_t self = getpid();
  srand(self);

  printf("pid: %i\n", self);

  char* check = NULL;
  double chance = strtod(argv[1], &check);
  if (check == argv[1] || errno == ERANGE)
    crash("error parsing argument as double: '%s'", argv[1]);
  char* text = argv[2];

  sleep(1);
  if (rand_double() < chance)
  {
    int rc = kill(self, SIGHUP);
    assert(rc == 0);
    sleep(1);
    printf("How am I alive?\n");
  }

  printf("text: %s\n", text);

  return 0;
}

static double rand_double(void)
{
  double r = (double) rand(); // ( (rand()+time(NULL)) % RAND_MAX );
  double m = (double) RAND_MAX;
  return r/m;
}

void
crash(char* fmt, ...)
{
  printf("child.x: crash: ");
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
  printf("\n");

  exit(EXIT_FAILURE);
}
