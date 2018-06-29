
#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void do_fork(const char* command);

static void vfail(const char* format, va_list va);


static void
check(bool condition, const char* format, ...)
{
  if (condition) return;

  va_list va;
  va_start(va, format);
  vfail(format, va);
  va_end(va);
}

static void
vfail(const char* format, va_list va)
{
  vprintf(format, va);
  printf("\n");
  exit(EXIT_FAILURE);
}

static void
fail(const char* format, ...)
{
  va_list va;
  va_start(va, format);
  vfail(format, va);
  va_end(va);
}
