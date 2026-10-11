#!/bin/sh
# Run from work.sql while the first shell holds a RESERVED lock (BEGIN IMMEDIATE): a second
# process's write must be refused, and its own message and status go to stdout for comparison.
"$1" "$2" "INSERT INTO t(name) VALUES ('other')" 2>&1
echo "second writer exit $?"
