
package require woztools

set results [ dict create ] 

while { [ gets stdin D ] >= 0 } {
  set filename $D/turbine.log
  set found [ grep "PROCS:" $filename L ]
  check $found "No PROCS in $filename !"
  assert { [ llength $L ] == 1 } "Expected single result!"
  set kv [ lindex $L 0 ]
  set tokens [ split $kv ":" ]
  # show tokens
  # puts [ llength $tokens ]
  set procs [ lindex $tokens 1 ]
  trim procs
  
  set filename $D/output.txt
  set found [ grep "MPIEXEC TIME:" $filename L ]
  check $found "No MPIEXEC TIME in $filename !"
  set kv [ lindex $L 0 ]
  set tokens [ split $kv ":" ]
  set t [ lindex $tokens 1 ]
  # Remove single quote:
  set t [ regsub "'" $t "" ]

  dict set results $procs $t
  # puts ""
}

set keys [ dict keys $results ]
set keys [ lsort -integer $keys ]
foreach k $keys {
  print $k [ dict get $results $k ]
}
