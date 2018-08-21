
# TD: Tcl Driver

TCL = ${HOME}/sfw/tcl-8.6.6

INCLUDES = -I $(TCL)/include 
LIBS = -L $(TCL)/lib -ltcl8.6

CPPFLAGS = $(INCLUDES)
CFLAGS = -fPIC -Wall

SHARED = -shared

libtd.so: td.o
	$(CC) $(SHARED) -o $(@) $(^) $(LIBS)

clean:
	rm -fv *.so *.o
