# mruby/release-differential on virtual Capstone, case 11 through the C API -- 2026-10-11

The 23 cases again, on the same two images, VM configuration and platform as
[2026-10-11-virtual](../2026-10-11-virtual/README.md): same mruby images, and every platform hash in
`inputs.json` equal except the runner's. One change: case 11 (`cb51fce92`) now runs its C-API
driver `capi.c` instead of `trigger.rb`, which at the 4.0.0-rc2 pin does not reach the defect (see
the case's `fidelity`). Runner: `probe/run-virtual.py`, which runs a case whose `trigger` is a `.c`
file as its own image, `capi-NN.dom` from `probe/build-capi.sh`, linked against the arm's own
`libmruby.a`.

| arm | mruby image | case 11 driver | control `smoke.rb` | fault | completed | time |
|---|---|---|---|---:|---:|---:|
| `virtual-malloc` | `7b5872bc7141cbad` (stock) | `261f758d08f5229a` | completed, `SMOKE_DONE` | 20 | 3 | 139 s |
| `virtual-nested-pools` | `321dc9b98691cc0a` (`MRBD_SUBLET=1`, patch 0008) | `09d77f573a6ed51d` | completed, `SMOKE_DONE` | 23 | 0 | 144 s |

Case 11 on both arms: `CASE11 ready`, then a capability fault, cause 25 (an invalid lifetime), in
`gc_mark_children`, one of the case's declared `fault_sites` (`matrix.tsv` columns `reached` and
`in_fault_sites`). The task's stack is freed with `mrb_free`, so it is musl mallocng's allocation,
and mallocng retires its lifetime on free: the stock arm catches it too, and patch 0008, which is
about GC object slots, adds nothing here. The other 22 rows are the earlier bundle's, cause and
function, on both arms.

## The driver reaches the defect: two matched pairs

Each pair differs by cb51fce92's `task.c` hunks alone, applied with `patch` to the port's tree: 4
of its 5 hunks apply, and the fifth, in `mrb_close_task`, has no target at the pin.

| build | run | result |
|---|---|---|
| native, pin (`probe/native-control.sh`) | 1 | `CASE11 ready`, SIGSEGV (exit 139) in `mrb_gc_mark` < `gc_mark_children` < `incremental_marking_phase` < `incremental_gc` < `incremental_gc_finish` < `mrb_full_gc` (gdb) |
| native, pin + fix | 1 | `CASE11 completed 30`, exit 0 |
| native ASan, pin (`NC_ASAN=1`) | 1 | `CASE11 ready`, then heap-use-after-free, READ of size 4, in `gc_mark_children` (gc.c:837), 8 bytes into a 1024-byte block freed by `mrb_free` in `mrb_execute_proc_synchronously` (task.c:1199) and allocated in `task_init_context` (task.c:267): the task's stack |
| native ASan, pin + fix | 1 | `CASE11 completed 30`, exit 0, no report |
| virtual-malloc, pin (`261f758d08f5229a`) | 2 | `CASE11 ready`, cause 25 in `gc_mark_children`, both runs |
| virtual-malloc, pin + fix (`bc758328cbab8b70`) | 2 | exit 0 both runs; `CASE11 completed 30` printed on one, the other's console lost that line |

The native backtrace is the one upstream's fix message reports (`mrb_gc_mark` under
`gc_mark_children`, the `MRB_TT_ENV` case, under `mrb_full_gc`). The virtual pair ran in one boot,
alternating. The fixed virtual build is `build-mruby-domain.sh` over a copy of the stock tree with
the hunks applied (79 s); each native pair is two full native builds with the port's native
configuration (127 s for both, with the runs; 144 s under ASan).

N = 1 per case cell in the matrix; N = 2 per arm for the virtual pair.
