# Tier 1 and Tier 2 records on the one-slot pin `852c5bc76774`

Two records the board's pre-registrations cite, both on the strict-DELIN / one-supervisor-slot pin
with the C-27-fixed images. Each was pre-registered before it ran, and each matched.

## Tier 1 — contract.dom's two single-context steps

```
T1_EXIT:ok:0        VIRTUAL_APPLICATION_OK present
T1_EXIT:stale:139   cause 25
T1_EXIT:cleanup:0
cause 24: 0   cause 26: 0   any `delin` mention: 0
geometry: node_bytes=1048576 node_capacity=65536
```

`ok` passes and `stale` faults with its own deliberate cause 25 — and **no cause 26**, which is the
C-27 fix holding on the pin that enforces DELIN strictly. No cause 24, because both steps are
single-context and never reach the slot limit.

  serial `<v8-kit>/t1-contract/guest/serial.log` (103,921 bytes), status `status-t1`

## Tier 2 — the three app steps, which closed an open caveat

These ran because of an ambiguity the Tier 3 count could not settle. `sqlite3.dom` and `mruby`
**link** `__capstone_vm_thread_create_frame` (sqlite is built `THREADSAFE=1`, so musl's pthread
layer is present), and linked is not called. In the 2x2 matrix those three steps had passed only on
the 32-slot pin; on the one-slot pin the run died at `threads` before reaching them, so they had no
reading on one slot at all. If any of them created a thread, the 39/41 figure for a one-slot machine
would be wrong and the boot-4 pre-registration would have inherited the error.

The gate's three invocations are **extracted verbatim from run.py's own SCRIPT**, not retyped, so
the record is comparable rather than merely similar; the extraction asserts exactly one match per
step.

```
T2_EXIT:sqlite_write:0    SQLITE_SUM=5050
T2_EXIT:sqlite_read:0     SQLITE_PERSIST=100
T2_EXIT:mruby:0           MRUBY_SUM=5050   MRUBY_FILE=virtual
T2_EXIT:cleanup:0
cause 24: 0   cause 26: 0   cause 29: 0   Scalar(0x1d): 0
```

  serial `<v8-kit>/t2-apps/guest/serial.log` (232,031 bytes), status `status-t2`,
  prediction `PREDICT-t2.txt`

**Reading.** None of the three app steps creates a domain thread. `__capstone_vm_thread_create_frame`
is linked into both images and never called — which is what a static symbol test could not decide
and this run settles behaviourally. So **39/41 stands** for a one-slot machine, with only `threads`
and `heap-threads` unavailable, and the serialization queue needs no app-step case.

## Inputs, both records

| input | value |
|---|---|
| QEMU pin | `capstone-qemu virtual-capstone-v7 852c5bc76774` |
| contract.dom | `f2017064bea77dd2` |
| sqlite3.dom | `233640f7193b034c` |
| mruby | `bba891bc16576d98` |
| launcher `capstone-vexec` | `a405840dce789345` |
| module (QEMU kernel, flat) | `c4440adde0d02347` |

The module is the QEMU-kernel flat adapter, **not** `ed24f827b5beb373`. They are the same source
built for two different kernels — the board's ko comes from the FPGA buildroot gcc — so
`c4440adde0d02347` is the correct QEMU-side counterpart and not a different revision. Both were
reused unchanged: C-27 is codegen and cannot touch either.
