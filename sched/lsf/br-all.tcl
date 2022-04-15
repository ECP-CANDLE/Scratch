
# BR ALL TCL

source $env(THIS)/font.tcl

set minnodes [ lindex $argv 0 ]
set maxnodes [ lindex $argv 1 ]

# Utilities

proc secs_to_mm:ss { secs mm ss } {
  upvar $mm m
  upvar $ss s
  set m [ expr int($secs / 60) ]
  set s [ expr $secs - ($m*60) ]
}

proc printf { args } {
    set newline ""
    if { [ lindex $args 0 ] == "-n" } {
        set newline "-nonewline"
        head args
    }
    if { [ llength $args ] == 0 } {
        error "printf: Requires format!"
    }
    set fmt [ lindex [ head args ] 0 ]
    puts {*}$newline [ format $fmt {*}$args ]
}

# Read headers from bjobs
gets stdin line

set results [ list ]

set empty_exit $env(EMPTY_EXIT)

# 2019-09-16 -> 4669
# 2021-03-31 -> 4728
# 2021-08-12 -> 4724
set total       4734
arrays nodes count
set nodes(busy) 0
set nodes(smll) 0
set nodes(bigs) 0
set nodes(mine) 0
set count(busy) 0
set count(smll) 0
set count(bigs) 0
set count(mine) 0

proc add { type n } {
  global count
  global nodes
  incr count($type)
  incr nodes($type) $n
}

set pending [ list ]

while { [ gets stdin line ] >= 0 } {

  # st: submit_time
  # 1st _ is for the "second(s)" token
  # 2nd _ is for the "L" token (hard runtime limit)
  # show .
  # show . line
  lassign $line jobid user stat queue hosts run_time _ time_left _ job_name

  if [ string is integer $hosts ] {
    # nexec_host/Summit adds an extra "batch" host
    incr hosts -1
  }

  secs_to_hh:mm $run_time h m
  if { $h eq "" } {
    set rt [ format "%2i" $m ]
  } else {
    set rt [ format "%i:%02i" $h $m ]
  }

  # show . -q stat

  if { $stat eq "RUN" } {
    if { $time_left eq "-" } {
      set tl "??"
    } else {
      set tl [ hm_to_hh:mm $time_left ]
    }
  } else {
    if { $user ne $env(USER) } continue
    set hosts 0
    set rt    "-"
    set tl    "--"
  }

  if { $user ne $env(USER) } {
    set job_name ""
  }

  set line [ format "%-6s  %-10s %4s %6s %5s %s" \
                 $jobid $user $hosts $rt $tl $job_name ]

  if { $stat eq "PEND" } {
    # show - pending line
    lappend pending $line
    continue
  }

  if { $user eq $env(USER) } {
    add mine $hosts
  }

  add busy $hosts
  if { $hosts < $minnodes } {
    add smll $hosts
    if { $user ne $env(USER) } {
      continue
    }
  } else {
    add bigs $hosts
    if { $hosts > $maxnodes && $user ne $env(USER) } {
      continue
    }
  }

  if { $queue eq "batch-hm" } {
    # Special queue we don't have to worry about:
    continue
  }

  # set time_left _${time_left}_
  if { [ regexp ".:.\\y" $time_left ] } {
    # show time_left
    append time_left "0"
  }

  if { $user eq $env(USER) } {
    set line [ bold $line ]
  }
  # show - result line
  lappend results [ list $run_time $line ]
}

if { [ llength $results ] == 0 } {
  if { $count(smll) == 0 } {
    puts "br-all.tcl: No results!"
    exit 1
  } else {
    puts "br-all.tcl: No jobs in range!"
  }
}

set results [ lsort -integer -index 0 -decreasing $results ]
set i 1
foreach result $results {
  lassign $result jobid info
  printf " %3i  %s" $i $info
  incr i
}

if { [ llength $pending ] > 0} {
  puts "      -------------------------------------"
  foreach line $pending {
    puts "      $line"
  }
}

set sp [ spaces 0 ]
set free [ expr $total - $nodes(busy) ]
set scale [ expr 100.0 / $total ]
set mine_pct [ expr round($scale * double($nodes(mine))) ]
set smll_pct [ expr round($scale * double($nodes(smll))) ]
set busy_pct [ expr round($scale * double($nodes(busy))) ]
set free_pct [ expr round($scale * double($free)) ]

puts ""
set counts "\[%3i\]"
foreach lbl [ list mine smll busy ] {
  set c [ format $counts $count($lbl) ]
  set pct [ expr round($scale * double($nodes($lbl))) ]
  printf "$sp %s:  %9s  %4i       %3i%%" \
      [ string toupper $lbl ] $c $nodes($lbl) $pct
}
printf "$sp FREE:             %4i/$total  %3i%%" $free        $free_pct

# printf "$sp BIGS:   %4i/$total" $bigs

if { $empty_exit && $nodes(mine) == 0 && [ llength $pending ] == 0 } {
  puts ""
  puts "You have no jobs running (empty-exit) !"
  exit 1
}
