# In-module exchange I/O

`micro.c` measures libc pipe reads, file reads and pipe writes at 1 byte,
4 KiB and 64 KiB. Build with the virtual SDK at `-O2`. Use the
`perf/linux-question-micro` branch's `run-micro.sh` with the output named
`lqmicro-cap`; provide its plain close helper and `lq-hog` as the runner requires.

For each mode and size run 20 and 220 rounds (2 and 12 at 64 KiB).
`python3 micro-table.py A=CONTROL_RUN B=ARM_RUN` reports instruction attempts
per call by layer, from the difference between the two windows divided by
the difference in rounds. Failed invocations and missing windows are errors.
The program checks byte counts and payload contents outside the measured window.
