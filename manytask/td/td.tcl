
load libtd.so

puts "TD START"

set rank [ c_init ]

set tasks 4

if { $rank == 0 } {
  c_serve $tasks
} else {
  while true {
    set cmd [ c_get ]
    puts "cmd: $cmd"
    if { $cmd eq "STOP" } break
    set cmd_list [ split $cmd " " ]
    exec {*}$cmd
  }
}

c_finalize 

puts "TD STOP"
