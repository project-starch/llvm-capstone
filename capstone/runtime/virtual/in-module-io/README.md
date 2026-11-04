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

The module serves only read/write/pread64/pwrite64 with the ABI-v2 offset
shape and a transport registered by the trusted launcher. Registration follows
signal attachment and remembers the metadata and exchange arena identities;
a forged exchange or an address recycled into another arena cannot opt in.
All other requests use the launcher. The launcher registers its private
file descriptors first; those and the module device remain inaccessible.

The metadata and exchange arenas are held while the kernel copies and the
VM lock is dropped. Retire, remap and conversion into a thread frame refuse
a held arena. The pending context cannot be stepped or destroyed during I/O.
The unchanged libc still copies back through the caller's capability, so an
object revoked while the read blocks faults at copy-back. No direct-buffer
request or libc shortcut is implemented.

A read interrupted before transferring bytes falls back to the launcher,
which publishes signals and retries under the existing signal ABI. A completed
call is marked for reply before any scheduling or signal stop, so it is not
served twice. `in_module_io=0` restores the launcher path for comparison.
`CV_TEST_HOOKS=1` alone exposes `inline_hold=0` and
`inline_test_transport=1`: the negative control can submit an arbitrary test
exchange and deliberately allow it to be retired during a blocked kernel copy.
Neither switch exists in the production build.
