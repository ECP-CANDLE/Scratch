
/**
   Simplest possible task distributor
 */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "stuff.h"

#define MODE_VANILLA    0 // No Tcl
#define MODE_TCL_LINK   1 // Link/init but nothing else
#define MODE_TCL_EXEC   2 // Use Tcl exec
#define MODE_TCL_SYSTEM 3 // Use Tcl extension for system()

#if MODE != MODE_VANILLA
#include <tcl.h>
#endif

#include <mpi.h>

static int rank, size;

#define buffer_size 1024
char buffer[buffer_size];

static const char* GET  = "GET";
static const char* STOP = "STOP";

static void master(int workers);
static void worker(void);

static void tcl_start(const char* program);
static void tcl_finalize(void);

#if MODE != MODE_VANILLA
Tcl_Interp* interp = NULL;
#endif

typedef enum { INPUT_UNKNOWN, INPUT_COUNT, INPUT_FILE } input_type;
static void parse_args(int argc, char** args);
static void report_settings(int size);

static int   task_count = -1;
static char* task_file = NULL;

int
main(int argc, char* argv[])
{
  MPI_Init(0, 0);

  double start = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  memset(buffer, 0, buffer_size);

  if (rank == 0)
  {
    parse_args(argc, argv);
    report_settings(size);
  }

  int workers = size-1;
  tcl_start(argv[0]);
  if (rank == 0)
    master(workers);
  else
    worker();

  double stop = MPI_Wtime();
  if (rank == 0)
    printf("TIME: %0.3f\n", stop-start);

  tcl_finalize();
  MPI_Finalize();
  return 0;
}

static void
parse_args(int argc, char** args)
{
  input_type input = INPUT_UNKNOWN;
  int opt;
  while ((opt = getopt(argc, args, "f:n:")) != -1)
  {
    switch (opt)
    {
      case 'f':
        check(input == INPUT_UNKNOWN, "provide only -f or -n");
        input = INPUT_FILE;
        task_file = optarg;
        break;
      case 'n':
        check(input == INPUT_UNKNOWN, "provide only -f or -n");
        input = INPUT_COUNT;
        int c = sscanf(optarg, "%i", &task_count);
        check(c == 1, "Could not parse as integer: %s", optarg);
        break;
    }
  }
  check(input != INPUT_UNKNOWN, "provide -f <file> or -n <count>");
}

static void report_settings(int size)
{
  if (rank == 0)
  {
    printf("MODE:  %i\n", MODE);
    printf("SIZE:  %i\n", size);
    if (task_file == NULL)
      printf("TASK COUNT: %i\n", task_count);
    else
      printf("TASK FILE: %s\n", task_file);
  }
}

static void distribute_from_count(void);
static void distribute_from_file(void);

static void
master(int workers)
{
  check(workers > 0, "No workers!");

  MPI_Status status;

  if (task_file != NULL)
    distribute_from_file();
  else
    distribute_from_count();

  for (int i = 0; i < workers; i++)
  {
    MPI_Recv(buffer, buffer_size, MPI_BYTE, MPI_ANY_SOURCE,
             0, MPI_COMM_WORLD, &status);
    strcpy(buffer, STOP);
    int worker = status.MPI_SOURCE;
    MPI_Send(buffer, buffer_size, MPI_BYTE, worker,
             0, MPI_COMM_WORLD);
  }
}

static void distribute_string(const char* task);
static char task[buffer_size];

static void
distribute_from_file()
{
  FILE* fp = fopen(task_file, "r");
  check(fp != NULL, "could not read: %s", task_file);
  while (fgets(task, buffer_size, fp) != NULL)
  {
    distribute_string(&task[0]);
  }
  fclose(fp);
}

static void
distribute_from_count()
{
  strcpy(task, "bash -c exit");
  for (int i = 0; i < task_count; i++)
  {
    distribute_string(&task[0]);
  }
}

static void
distribute_string(const char* task)
{
  MPI_Status status;
  MPI_Recv(buffer, buffer_size, MPI_BYTE, MPI_ANY_SOURCE,
           0, MPI_COMM_WORLD, &status);

  int worker = status.MPI_SOURCE;
  int length = strlen(task)+1;
  MPI_Send(task, length, MPI_BYTE, worker, 0, MPI_COMM_WORLD);
}

static int execute(const char* command);

static void
worker()
{
  int count = 0;
  MPI_Status status;
  while (true)
  {
    printf("GET\n");
    strcpy(buffer, GET);
    MPI_Send(buffer, buffer_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(buffer, buffer_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
             &status);
    if (strcmp(buffer, STOP) == 0)
      break;
    int rc = execute(buffer);
    if (rc != 0)
      printf("command failed on rank: %i : %s\n", rank, buffer);
    count++;
  }
  // printf("worker rank: %i : tasks: %i\n", rank, count);
}

static
void tcl_start(const char* program)

#if MODE == MODE_VANILLA
// No Tcl
{}
#else
{
  Tcl_FindExecutable(program);
  interp = Tcl_CreateInterp();
  int rc = Tcl_Init(interp);
  check(rc == TCL_OK, "Tcl_Init failed!");
}
#endif

static int
execute(const char* command)
#if MODE == MODE_VANILLA || MODE == MODE_TCL_LINK
{
  printf("execute: %s\n", command);
  do_fork(command);
}
#elif MODE == MODE_TCL_EXEC
{
  char script[buffer_size];
  sprintf(script, "exec %s", command);
  int rc = Tcl_Eval(interp, script);
  if (rc != TCL_OK)
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
#elif MODE == MODE_TCL_SYSTEM



#endif


static
void tcl_finalize()
#if MODE == MODE_VANILLA
// No Tcl
{}
#else
{
  Tcl_Finalize();
}
#endif


// UNUSED STUFF FOLLOWS

#if 0
const int buffer_size = 1024;
char buffer[buffer_size];

#define assert_msg(condition, format, args...)  \
    { if (!(condition))                          \
       assert_msg_impl(format, ## args);        \
    }

/**
   We bundle everything into one big printf for MPI
 */
void
assert_msg_impl(const char* format, ...)
{
  char buffer[buffer_size];
  int count = 0;
  char* p = &buffer[0];
  va_list ap;
  va_start(ap, format);
  count += sprintf(p, "error: ");
  count += vsnprintf(buffer+count, (size_t)(buffer_size-count), format, ap);
  va_end(ap);
  printf("%s\n", buffer);
  fflush(NULL);
  exit(EXIT_FAILURE);
}
#endif
