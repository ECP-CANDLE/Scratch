
# MODE:

MODE_VANILLA    = 0 # No Tcl
MODE_TCL_LINK   = 1 # Link/init but nothing else
MODE_TCL_EXEC   = 2 # Use Tcl exec
MODE_TCL_SYSTEM = 3 # Use Tcl extension for system()

# USER EDIT START

MODE = $(MODE_TCL_SYSTEM)

# Theta:
TCL = ${HOME}/Public/sfw/theta/tcl-8.6.6-mt-g

# Dunedin:
# MODE = $(MODE_VANILLA)
# TCL = ${HOME}/Public/sfw/x86_64/tcl-8.6.6-global-gcc-4.8.1
# TCL = ${HOME}/sfw/tcl-8.6.8-global

# USER EDIT END

ifneq ($(MODE),$(MODE_VANILLA))
	INCLUDES = -I $(TCL)/include
	LIBS = -L $(TCL)/lib -l tcl8.6 -Wl,-rpath -Wl,$(TCL)/lib
endif

CPPFLAGS = $(INCLUDES)
CFLAGS   = -DMODE=$(MODE) -fPIC

all: manytask.x

%.o: Makefile stuff.h

once.x: once.o stuff.o
	$(CC) -o $(@) $(^) $(LIBS)

manytask.x: main.o stuff.o
	$(CC) -o $(@) $(^) $(LIBS)

clean::
	rm -fv *.x *.o
