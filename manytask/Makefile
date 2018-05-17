
# MODE:
# 0 = vanilla, no Tcl
# 1 = just Tcl linkage
# 2 = Tcl exec
# 3 = Tcl -> system()

CFLAGS = -DMODE=1

TCL = ${HOME}/Public/sfw/x86_64/tcl-8.6.6-global-gcc-4.8.1

LIBS = -L $(TCL)/lib -l tcl8.6

all: manytask.x

main.o: Makefile

manytask.x: main.o
	$(CC) -o $(@) $(<) $(LIBS)

clean::
	rm -fv *.x *.o
