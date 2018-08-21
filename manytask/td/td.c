
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <mpi.h>
#include <tcl.h>

#define buffer_size 1024
static char buffer[buffer_size];

static const char* GET  = "GET";
static const char* STOP = "STOP";

static int mpi_rank, mpi_size;
static double time_start;

static int
c_init(ClientData cdata, Tcl_Interp *interp,
       int objc, Tcl_Obj *const objv[])
{
  memset(buffer, 0, buffer_size);
  
  MPI_Init(0, 0);

  time_start = MPI_Wtime();
  
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  pid_t self = getpid();
  printf("rank: %i self: %i\n", mpi_rank, self);

  char filename[1024];
  sprintf(filename, "out-%04i.txt", mpi_rank);
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
  char** cmd = malloc(objc * sizeof(char*));
  for (int i = 0; i < objc; i++)
    cmd[i] = strdup(Tcl_GetStringFromObj(objv[1], NULL));
  pid_t pid = fork();
  if (pid == 0)
  {
    execvp(cmd[0], cmd);
    printf("execvp failed!\n");
  }
  else
  {
    int status;
    int rc = waitpid(pid, &status, 0);
    assert(rc > 0);
    if (WIFEXITED(status))
    {
      int exitcode = WEXITSTATUS(status);
      if (exitcode != 0)
        printf("Child exited with code: %i", exitcode);
    }
    else if (WIFSIGNALED(status))
    {
      int sgnl = WTERMSIG(status);
      printf("Child pid=%i killed by signal: %i", pid, sgnl);
    }
    else
    {
      printf("TURBINE: UNKNOWN ERROR in pid_status()\n");
      exit(1);
    }
  }

  printf("Child OK\n");
  for (int i = 0; i < objc; i++)
    free(cmd[i]);
  free(cmd);

  return TCL_OK;
}

static int
c_finalize(ClientData cdata, Tcl_Interp *interp,
           int objc, Tcl_Obj *const objv[])
{
  MPI_Barrier(MPI_COMM_WORLD);
  double time_stop = MPI_Wtime();
  if (mpi_rank == 0)
    printf("TIME: %0.3f\n", time_stop-time_start);
  MPI_Finalize();
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
