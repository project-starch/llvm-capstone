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
Workers using the shared process transport bracket their launcher serve with
`CV_IO_HOST` while owning `host_lock`. This waits for an in-flight main inline
call and makes later main requests use the launcher lock until the serve ends.
Private worker transports remain independent. This extends the shared-host
serialization to cover the module as well as native workers.

The metadata and exchange arenas are held while the kernel copies and the
VM lock is dropped. Retire, remap and conversion into a thread frame refuse
a held arena. The pending context cannot be stepped or destroyed during I/O.
The unchanged libc still copies back through the caller's capability, so an
object revoked while the read blocks faults at copy-back. No direct-buffer
request or libc shortcut is implemented.

A read interrupted before transferring bytes reports its Linux restart reason
as a service-event cause. After the Linux signal trampoline records the event,
the launcher publishes DONE/EINTR without SA_RESTART, or RETRY with it.
Blindly returning to the syscall stub would misclassify an interrupted call as
not yet submitted and restart it even without SA_RESTART. A completed
call is marked for reply before any scheduling or signal stop, so it is not
served twice. `in_module_io=0` restores the launcher path for comparison.
`CV_TEST_HOOKS=1` alone exposes `inline_hold=0` and
`inline_test_transport=1`: the negative control can submit an arbitrary test
exchange and deliberately allow it to be retired during a blocked kernel copy.
Neither switch exists in the production build.


## Directed contract

Build `contract.c` with the arm's virtual SDK at `-O2 -lpthread`, adding the
runtime include directory. Build `observe.c` with the native RISC-V Linux
compiler at `-O2 -static -Wall -Wextra -Werror`. Both binaries and OUT must be
below SHARE, a private lane directory. Use a fresh OUT for each run:

```
python3 run-contract.py --platform PLATFORM --host TREE/capstone/runtime/host \
  --domain contract.dom --observer observe --share SHARE --out OUT
```

The driver runs functional file/pipe/positional calls, interrupted reads with
and without SA_RESTART, revocation during a blocked libc read and before
submission, and invalid exchange claims. Each revoked-buffer process must
terminate with cause 25. The native observer keeps a pipe endpoint across each
refused domain process and verifies that all 256 input bytes remain. The span
case crosses two adjacent grants; the other claims use an unmapped address,
image text, a retired grant and an unregistered live transport.

Run again with a `CV_TEST_HOOKS=1` module and `--test-hooks`. This explicitly
enables arbitrary test exchanges to exercise the arena checks and holds.
With the hold on, a blocked exchange read prevents unmap (EBUSY), changes no
replacement arena and releases the hold after completion. With the hold off,
the same test must recycle the address and land all 256 bytes in its replacement.
The negative control must print LANDED; its absence is a failure. The normal
module must not expose either test parameter. `result.json` records every
expected exit and marker; a missing case, assertion, timeout or observer
mismatch prevents a passing result. No profiling window is required here.

`host-contract.c` includes the real launcher and supplies a blocking module
stand-in. Link it with the same native Linux/common sources as
`test-transport-host.sh`, using `-O1 -pthread` and the virtual/runtime include
paths. It verifies CV_IO_HOST acquisition, serve and release, and that a private
service can make progress while acquisition waits. Also run
`test-transport-host.sh` and `test-detach-deadlock.sh` after integration changes.
