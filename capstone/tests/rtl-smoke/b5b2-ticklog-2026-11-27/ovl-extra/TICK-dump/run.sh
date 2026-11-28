#!/bin/sh
# tick-log dump (runtime lane, b5b2-ticklog-2026-11-27): freeze the table, print its sha256, then it gzip'ed in base64.
echo 0 > /proc/ticklog || { echo "@@TICKLOG-ERROR freeze failed"; exit 1; }
echo "@@TICKLOG-SHA $(sha256sum < /proc/ticklog | cut -d' ' -f1) bytes=$(wc -c < /proc/ticklog)"
echo "@@TICKLOG-BEGIN"
gzip -9c < /proc/ticklog | base64
echo "@@TICKLOG-END"
