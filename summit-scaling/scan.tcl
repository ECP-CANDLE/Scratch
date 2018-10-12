
proc abort { msg } {
  puts $msg
  exit 1
}

# Shutdown Tcl if condition does not hold
proc check { condition msg } {
    if { ! [ uplevel 1 "expr $condition" ] } {
      abort $msg
    }
}

proc assert { condition msg } {
    if { ! [ uplevel 1 "expr $condition" ] } {
        error $msg
    }
}

proc trim { s* } {
  upvar ${s*} s
  set s [ string trim $s ]
}

proc print { args } {
  set n [ llength $args ]
  for { set i 0 } { $i < $n } { incr i } {
    puts -nonewline [ lindex $args $i ]
    if { $i < $n - 1 } { puts -nonewline " " } 
  }
  puts ""
}

# Return true if we find anything, matches in list L
proc grep { pattern filename L* } {
  upvar ${L*} L
  set result false
  set L [ list ]
  set fd [ open $filename "r" ]
  while { [ gets $fd line ] >= 0 } { 
    if [ regexp $pattern $line ] {
      lappend L $line
      set result true
    }
  }
  close $fd
  return $result
}

# usage: head <list> <value>?
# @return and remove first element of list
proc head { args } {
    set count 1
    if { [ llength $args ] == 1 } {
        set name [ lindex $args 0 ]
    } elseif { [ llength $args ] == 2 } {
        set count [ lindex $args 0 ]
        set name  [ lindex $args 1 ]
    } else {
        error "head: requires: <count>? <list> - received: $args"
    }

    upvar $name L

    set result [ list ]
    for { set i 0 } { $i < $count } { incr i } {
        lappend result [ lindex $L 0 ]
        set L [ lreplace $L 0 0 ]
    }
    return $result
}

set D [ head argv ]

set filename $D/turbine.log

set found [ grep "JOB:" $filename L ]
check $found "No JOB in $filename !"
assert { [ llength $L ] == 1 } "Expected single result!"
set kv [ lindex $L 0 ]
set tokens [ split $kv ":" ]
# show tokens
# puts [ llength $tokens ]
set job [ lindex $tokens 1 ]
trim job

set found [ grep "PROCS:" $filename L ]
check $found "No PROCS in $filename !"
assert { [ llength $L ] == 1 } "Expected single result!"
set kv [ lindex $L 0 ]
set tokens [ split $kv ":" ]
# show tokens
# puts [ llength $tokens ]
set procs [ lindex $tokens 1 ]
trim procs

set elapsed ""
foreach filename $argv { 
  
  set found [ grep "Elapsed" $filename L ]
  if { ! $found } continue
  set kv [ lindex $L 0 ]
  set tokens [ split $kv ":" ]
  set t [ lindex $tokens 1 ]
  # Remove single quote:
  # set t [ regsub "'" $t "" ]
  set elapsed $t
  # puts ""
}

if { [ string length $elapsed ] == 0 } {
  puts "No 'Elapsed' in $argv !"
  exit 1
}

puts "$job $procs $elapsed"
