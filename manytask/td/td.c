
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <mpi.h>
#include <tcl.h>

// #include <data.h>
// #include <tools.h>

#define buffer_size 1024
static char buffer[buffer_size];

static const char* GET  = "GET";
static const char* STOP = "STOP";

static int mpi_rank, mpi_size;
static double time_start;

static void print(const char* format, ...);

static int
c_init(ClientData cdata, Tcl_Interp *interp,
       int objc, Tcl_Obj *const objv[])
{
  memset(buffer, 0, buffer_size);

  MPI_Init(0, 0);

  time_start = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);

  pid_t self = getpid();
  print("rank: %5i self: %5i host: %10s\n",
         mpi_rank, self, hostname);

  // time_delay(0.1);
  // xlb_data_init(1,0);

  char filename[1024];
  sprintf(filename, "rank-%04i.txt", mpi_rank);
  freopen(filename, "w", stdout);

  Tcl_Obj* result = Tcl_NewIntObj(mpi_rank);
  Tcl_SetObjResult(interp, result);

  return TCL_OK;
}

char* task = "echo HELLO";

static int
c_serve(ClientData cdata, Tcl_Interp *interp,
        int objc, Tcl_Obj *const objv[])
{
  assert(objc == 2);

  int tasks;
  Tcl_GetIntFromObj(interp, objv[1], &tasks);

  int workers = mpi_size-1;

  MPI_Status status;
  for (int i = 0; i < tasks; i++)
  {
    MPI_Recv(buffer, buffer_size, MPI_BYTE, MPI_ANY_SOURCE,
             0, MPI_COMM_WORLD, &status);

    int worker = status.MPI_SOURCE;
    int length = strlen(task)+1;
    MPI_Send(task, length, MPI_BYTE, worker, 0, MPI_COMM_WORLD);
  }

  for (int i = 0; i < workers; i++)
  {
    MPI_Recv(buffer, buffer_size, MPI_BYTE, MPI_ANY_SOURCE,
             0, MPI_COMM_WORLD, &status);
    strcpy(buffer, STOP);
    int worker = status.MPI_SOURCE;
    MPI_Send(buffer, buffer_size, MPI_BYTE, worker,
             0, MPI_COMM_WORLD);
  }
  return TCL_OK;
}

static int
c_get(ClientData cdata, Tcl_Interp *interp,
      int objc, Tcl_Obj *const objv[])
{
  strcpy(buffer, GET);
  MPI_Send(buffer, buffer_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
  MPI_Status status;
  MPI_Recv(buffer, buffer_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
           &status);

  Tcl_Obj* result = Tcl_NewStringObj(buffer, -1);
  Tcl_SetObjResult(interp, result);
  return TCL_OK;
}

static int
c_system(ClientData cdata, Tcl_Interp *interp,
         int objc, Tcl_Obj *const objv[])
{
  print("c_system\n");
  char** cmd = malloc(objc * sizeof(char*));
  int i = 0;
  for ( ; i < objc; i++)
    cmd[i] = strdup(Tcl_GetStringFromObj(objv[i+1], NULL));
  cmd[i] = NULL;
  print("FORK\n");
  pid_t pid = fork();
  if (pid == 0)
  {
    execvp(cmd[0], cmd);
    print("FAIL:  execvp failed!\n");
    print("ERROR: %s\n", strerror(errno));
  }
  else
  {
    int status;
    int rc = waitpid(pid, &status, 0);
    assert(rc > 0);
    if (WIFEXITED(status))
    {
      int exitcode = WEXITSTATUS(status);
      print("EXITED\n");
      if (exitcode != 0)
        print("FAIL: Child exited with code: %i", exitcode);
    }
    else if (WIFSIGNALED(status))
    {
      int sgnl = WTERMSIG(status);
      print("FAIL: Child pid=%i killed by signal: %i", pid, sgnl);
    }
    else
    {
      print("FAIL: UNKNOWN ERROR in pid_status()\n");
      exit(1);
    }
  }

  print("Child OK\n");
  /* for (int i = 0; i < objc-1; i++) */
  /*   free(cmd[i]); */
  /* free(cmd); */

  return TCL_OK;
}

static int
c_finalize(ClientData cdata, Tcl_Interp *interp,
           int objc, Tcl_Obj *const objv[])
{
  print("BARRIER\n");
  MPI_Barrier(MPI_COMM_WORLD);
  double time_stop = MPI_Wtime();
  if (mpi_rank == 0)
    print("TIME: %0.3f\n", time_stop-time_start);
  MPI_Finalize();
  print("FINALIZED\n");
  return TCL_OK;
}

int
Td_Init(Tcl_Interp* interp)
{
  if (Tcl_InitStubs(interp, "8.6", 0) == NULL)
    return TCL_ERROR;

  Tcl_CreateObjCommand(interp, "c_init",     c_init,     NULL, NULL);
  Tcl_CreateObjCommand(interp, "c_serve",    c_serve,    NULL, NULL);
  Tcl_CreateObjCommand(interp, "c_get",      c_get,      NULL, NULL);
  Tcl_CreateObjCommand(interp, "c_system",   c_system,   NULL, NULL);
  Tcl_CreateObjCommand(interp, "c_finalize", c_finalize, NULL, NULL);

  return TCL_OK;
}

static void
print(const char* format, ...)
{
  va_list va;
  va_start(va, format);
  vprintf(format, va);
  va_end(va);
  fflush(stdout);
}
