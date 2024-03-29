
proc bold { s } {
  return "\033\[1m${s}\033\[0m"
}

set colors [ dict create           \
             "Black"        "0;30" \
             "Red"          "0;31" \
             "Green"        "0;32" \
             "Brown/Orange" "0;33" \
             "Blue"         "0;34" \
             "Purple"       "0;35" \
             "Cyan"         "0;36" \
             "Light Gray"   "0;37" \
             "Dark Gray"    "1;30" \
             "Light Red"    "1;31" \
             "Light Green"  "1;32" \
             "Yellow"       "1;33" \
             "Light Blue"   "1;34" \
             "Light Purple" "1;35" \
             "Light Cyan"   "1;36" \
             "White"        "1;37" \
            ]

proc color { color s } {
  global colors
  set c [ dict get $colors $color ]
  return "\033\[0$c${s}\033\[0m"
}
