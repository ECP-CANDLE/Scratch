
# MODE:

MODE_VANILLA    = 0 # No Tcl
MODE_TCL_LINK   = 1 # Link/init but nothing else
MODE_TCL_EXEC   = 2 # Use Tcl exec
MODE_TCL_SYSTEM = 3 # Use Tcl extension for system()

# USER EDIT START
MODE = $(MODE_VANILLA)
# TCL = ${HOME}/Public/sfw/x86_64/tcl-8.6.6-global-gcc-4.8.1
# TCL = ???
# USER EDIT END

ifneq ($(MODE),$(MODE_VANILLA))
	LIBS = -L $(TCL)/lib -l tcl8.6
endif

CFLAGS = -DMODE=$(MODE)

all: manytask.x

main.o: Makefile

manytask.x: main.o
	$(CC) -o $(@) $(<) $(LIBS)

clean::
	rm -fv *.x *.o
