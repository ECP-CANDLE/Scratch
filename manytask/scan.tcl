
package require woztools

set results [ dict create ]

proc grep_kv { token results* filename } {

  upvar ${results*} results
  set results [ list ]

  set found [ grep "$token:" $filename L ]
  show found
  if { ! $found } { return false }

  set kv [ lindex $L 0 ]
  set tokens [ split $kv ":" ]
  set result [ lindex $tokens 1 ]
  trim result
  lappend results $result
}

while { [ gets stdin D ] >= 0 } {

  set data [ list ]

  set filename $D/turbine.log
  foreach k { "JOB" "PROCS" } {
    set found [ grep_kv $k L $filename ]
    assert { [ llength $L ] == 1 } \
        "Expected single result for $k! found: [llength $L]"
    lappend data $k $L
  }

  set filename $D/output.txt
  set found [ grep_kv "MPIEXEC TIME:" L $filename ]
  assert { [ llength $L ] == 1 } \
      "Expected single result for MPIEXEC TIME! found: [llength $L]"
  lappend data "TIME" $L

  lappend results $data
  # puts ""
}

foreach result $results {
  puts $result
}
