#!/bin/zsh

XS=( 750 746 757 771 743
     744 759 763 )

CP=$HOME/proj/SV/workflows/cp-leaveout
ROOT=$CP/experiments

(
  sw0
  date-nice
  renice -n 19 $$
  print

  for X in $XS
  do
    D=$ROOT/X$X
    cd $D
    print X$X
    print
    last-access -n 3 .
    touch-all .
    print
  done
  date-nice
  sw1
) |& teeb prevent-delete.log
