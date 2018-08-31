#!/bin/zsh
set -eu

DIR=$1

lsd_leaf $DIR | tclsh scan.tcl
