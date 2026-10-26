#!/bin/bash
# runs 2..50 per arm, interleaved, alternating which arm goes first
cd $HOME/spec-cap/gates-1018/PT50-250
echo "start $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg)" >> results.txt
for n in $(seq 2 50); do
  if [ $((n % 2)) -eq 0 ]; then order="sdev sreg"; else order="sreg sdev"; fi
  for a in $order; do ./one.sh $a $n >> results.txt 2>&1; done
done
echo "done $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg)" >> results.txt
