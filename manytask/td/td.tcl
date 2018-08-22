
load libtd.so

puts "TD START"

set rank [ c_init ]

set tasks [ lindex $argv 0 ]

# source $env(THIS)/tcl/master.tcl

if { $rank == 0 } {
  puts "serving tasks: $tasks"
  c_serve $tasks
} else {
  while true {
    set cmd [ c_get ]
    puts "cmd: $cmd"
    if { $cmd eq "STOP" } break
    set cmd_list [ split $cmd " " ]
    # exec {*}$cmd
    puts "system:"
    c_system {*}$cmd_list
  }
}

c_finalize

puts "TD STOP"
