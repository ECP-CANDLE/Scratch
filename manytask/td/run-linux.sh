#!/bin/sh

. ./settings-linux.sh

export THIS=$( readlink --canonicalize-existing $( dirname $0 ) )

export LD_LIBRARY_PATH=$THIS

mpiexec -l -n 2 $TCL/bin/tclsh8.6 td.tcl 8
