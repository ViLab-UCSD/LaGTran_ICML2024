#!/bin/bash
classes=("4" "9" "19" "39" "79")
for ((i=0;i<${#classes[@]};++i)); do

    j = $((i+1))
    # line_no=`grep -n " $i$" real.txt| tail -1 | awk -F ":" '{print $1}'`
    # cat real.txt | head -$line_no > real_$j.txt
    echo $i
    echo $j

done