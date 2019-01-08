
# TD: Tcl Driver

# TCL = ${HOME}/sfw/tcl-8.6.6

SWIFT  = ${HOME}/proj/swift-t-2018-08-12
CUTILS = $(SWIFT)/c-utils/code
ADLB   = $(SWIFT)/lb/code

INCLUDES = -I ${TCL}/include ${EXTRA_INCLUDES}
# -I $(CUTILS)/src -I $(ADLB)/src -I $(ADLB)
LIBS =  -L $(TCL)/lib  -ltcl8.6
#	-L $(ADLB)/lib -ladlb   \
#	-L $(CUTILS)/lib -lexmcutils
#RPATHS = -Wl,-rpath -Wl,$(ADLB)/lib   \
#         -Wl,-rpath -Wl,$(CUTILS)/lib

CPPFLAGS = $(INCLUDES)
CFLAGS = -fPIC -Wall -g

SHARED = -shared

all: libtd.so

td.o: Makefile

libtd.so: td.o
	$(CC) $(SHARED) -o $(@) $(^) \
		$(LIBS) \
		$(RPATHS)

clean:
	rm -fv *.so *.o

.PHONY: clean
