
#include <assert.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "stuff.h"

static void pid_status(pid_t child);

const int MAX_ARGS = 16;

void do_fork(const char* command)
{
  // printf("do_fork()\n");
  char* cmd = strdup(command);
  pid_t child = fork();
  if (child == -1)
    fail("could not fork!");
  if (child == 0)
  {
    // printf("in child\n");
    char* t[MAX_ARGS];
    int i = 0;
    char* p = strtok(cmd, " ");
    while (true)
    {
      if (p == NULL) break;
      t[i] = strdup(p);
      // printf("t[%i]: %s\n", i, t[i]);
      i++;
      p = strtok(NULL, " ");
    }
    t[i] = NULL;
    int rc = execvp(cmd, t);
    printf("child failed! %s\n", strerror(errno));
    exit(0);
  }
  // printf("child: %i\n", child);
  pid_status(child);
  free(cmd);
}
    
static void pid_status(pid_t child)
{
  int rc;
  int status;
  char message[1024];
  rc = waitpid(child, &status, 0);
  assert(rc > 0);
  if (WIFEXITED(status))
  {
    int exitcode = WEXITSTATUS(status);
    if (exitcode != 0)
      printf(message,
             "Child exited with code: %i", exitcode);
  }
  else if (WIFSIGNALED(status))
  {
    int sgnl = WTERMSIG(status);
    printf(message, "Child killed by signal: %i", sgnl);
  }
  else
  {
    printf("UNKNOWN ERROR in pid_status()\n");
    exit(1);
  }
}
