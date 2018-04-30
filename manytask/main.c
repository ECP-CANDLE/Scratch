
/**
   Simplest possible task distributor
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

int rank, size;

#define buffer_size 1024
char buffer[buffer_size];

const char* GET  = "GET";
const char* STOP = "STOP";

static void check(bool condition, const char* format, ...);
static void fail(const char* format, va_list va);

static void master(int n, int workers);
static void worker(void);

int
main(int argc, char* argv[])
{
  MPI_Init(0, 0);

  double start = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  check(argc == 2, "Requires task count!");

  int n;
  int c = sscanf(argv[1], "%i", &n);
  check(c == 1, "Could not parse as integer: %s", argv[1]);

  memset(buffer, 0, buffer_size);

  int workers = size-1;

  if (rank == 0)
    master(n, workers);
  else
    worker();

  double stop = MPI_Wtime();
  if (rank == 0)
    printf("TIME: %0.3f\n", stop-start);

  MPI_Finalize();
  return 0;
}

void
master(int n, int workers)
{
  MPI_Status status;
  for (int i = 0; i < n; i++)
  {
    MPI_Recv(buffer, buffer_size, MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    strcpy(buffer, "bash -c exit");
    int worker = status.MPI_SOURCE;
    MPI_Send(buffer, buffer_size, MPI_BYTE, worker, 0, MPI_COMM_WORLD);
  }
  for (int i = 0; i < workers; i++)
  {
    MPI_Recv(buffer, buffer_size, MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    strcpy(buffer, STOP);
    int worker = status.MPI_SOURCE;
    MPI_Send(buffer, buffer_size, MPI_BYTE, worker, 0, MPI_COMM_WORLD);
  }
}

void
worker()
{
  int count = 0;
  MPI_Status status;
  while (true)
  {
    strcpy(buffer, GET);
    MPI_Send(buffer, buffer_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(buffer, buffer_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
    if (strcmp(buffer, STOP) == 0)
      break;
    system(buffer);
    count++;
  }
  printf("worker: %i\n", count);
}

static void
check(bool condition, const char* format, ...)
{
  if (condition) return;

  va_list va;
  va_start(va, format);
  fail(format, va);
  va_end(va);
}

static void
fail(const char* format, va_list va)
{
  if (rank == 0)
  {
    vprintf(format, va);
    printf("\n");
  }
  exit(EXIT_FAILURE);
}
