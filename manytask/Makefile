
# MODE:

MODE_VANILLA    = 0 # No Tcl
MODE_TCL_LINK   = 1 # Link/init but nothing else
MODE_TCL_EXEC   = 2 # Use Tcl exec
MODE_TCL_SYSTEM = 3 # Use Tcl extension for system()

MODE = MODE_VANILLA

CFLAGS = -DMODE=$(MODE)

TCL = ${HOME}/Public/sfw/x86_64/tcl-8.6.6-global-gcc-4.8.1

LIBS = -L $(TCL)/lib -Bdynamic -l tcl8.6

all: manytask.x

main.o: Makefile

manytask.x: main.o
	$(CC) -o $(@) $(<) $(LIBS)

clean::
	rm -fv *.x *.o
