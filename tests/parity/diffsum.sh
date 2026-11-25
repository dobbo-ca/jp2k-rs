#!/bin/sh
# diffsum.sh A B: succeeds if the trees are equal; otherwise prints the number of differences and
# about 60 of them spread over the sorted list, with both contents, and fails.
diff -rq "$1" "$2" >/tmp/diff.txt && exit 0
n=$(wc -l </tmp/diff.txt)
echo "$n differences"
awk -v step=$(( n / 60 + 1 )) 'step == 1 || NR % step == 1' /tmp/diff.txt | while read -r kind a rest; do
  if [ "$kind" = Files ]; then
    b=${rest%% differ*}; b=${b#and }
    echo "== $a"; head -c 300 "$a"; echo "-- other"; head -c 300 "$b"
  else
    # Only in DIR: NAME. One side has NAME.out and the other NAME.err: show both.
    dir=${rest%%:*}; file=${rest#*: }; name=${file%.*}
    if [ "$dir" = "$1" ]; then other=$2; else other=$1; fi
    echo "== $dir/$file"; head -c 300 "$dir/$file"; echo "-- other"
    for e in out err; do [ -f "$other/$name.$e" ] && head -c 300 "$other/$name.$e"; done
  fi
done
exit 1
