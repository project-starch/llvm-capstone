# mimalloc-bench on virtual Capstone

The Capstone side of the malloc-interface comparison with default CheriBSD
(`experiments/cheribsd-malloc-quarantine`, `capstone/experiments/malloc-quarantine/`).
The question is what each system's temporal protection costs at `malloc`/`free`:
memory per byte the program holds, and how soon a freed address is handed out again.

| System | Allocator | Protection |
|---|---|---|
| CheriBSD (default) | jemalloc behind MRS | freed memory waits in a quarantine until an asynchronous revocation sweep (ratio 1/4) |
| Virtual Capstone | musl mallocng compiled for Capstone, in the program; object capabilities through the Sublet primitives (`runtime/virtual/heap.c`) | `free` ends the object's lifetime at once; the slot can be reused immediately |
| Native musl (reference) | the same mallocng, x86-64, static | none |
| CheriBSD `off` (reference) | jemalloc, MRS revocation disabled | none |

QEMU time is not a result. Everything reported is counted: bytes, pages, allocations,
lifetime IDs, sweep passes.

## Programs

The fixed-work C programs of mimalloc-bench 69c41ed, with its arguments and one thread:

| Program | Arguments | Allocations (native) |
|---|---|---|
| barnes | `< barnes.input` | 18 |
| mleak | `5`, `50` | 5,500 / 55,000 |
| mstress | `1 50 25` | 187,575 |
| espresso | `largest.espresso` | 33,509,500 |
| cfrac | `17545186520507317056371138836327483792789528` | 91,530,283 |
| glibc-simple | | 96,000,003 |
| sh6bench | `1` | 193,000,001 |

The allocation counts are the program's, not the allocator's: CheriBSD's tracer counts the
same numbers (cfrac 91,530,284, glibc-simple 96,000,004, espresso 33,509,501, sh6bench
193,000,001; the extra one is an allocation made before the tracer starts). A run's progress
is therefore its `op=` over these totals.

Not run: sh8bench, which stores a pointer into an 8-byte block (`sh8bench-new.c:347-352`):
with 16-byte capabilities the store overflows the block, and both capability systems stop it
(Capstone: cause 28 after about a minute, in all four runs of 2026-10-09; CheriBSD's `off` arm
aborted abnormally after 247 s, not localized further). It runs only natively (437,061,564 allocations). glibc-thread and xmalloc-test (a fixed time, not a fixed amount of work); rptest
(needs a capability compare-and-swap through `__sync_*` and pointer provenance through
`uintptr_t`, see `build-domain.sh`); the C++ programs.

## What each run records

Every Capstone run prints only to the serial console. `gate.sh` streams everything a run
writes as it happens, so a long run shows its progress and a run that dies keeps every
sample it took; `MB_OUT_LINES` at the end lets a parser check the stream is complete.

| Line | When | Fields |
|---|---|---|
| `MB_RUSAGE` | end | the launcher process's peak RSS (its `mm` holds the program), user/system time, status |
| `CAPSTONE_VM_SAMPLE` | every second of launcher time | adapter counters: arenas, pinned pages and their peak, nodes minted/live/retired/high water, node table capacity and bytes, growths, collections, reclaimed IDs, page faults, steps; process VM size and RSS |
| `CAPSTONE_VM_STATS` | end | the same counters once more, plus delegated rounds and bytes, launch and elapsed time |
| `MB_OUT_LINES`, `MB_OUT_BYTES`, `MB_OUT_SHA256` | end | lines the run wrote; size and hash of the program's own output, compared with the native run |

Traced builds (`*-traced.dom`, `mqtrace-cap.c` linked through `--wrap`) add:

| Line | When | Fields |
|---|---|---|
| `MQ op=` | every 4096 allocations | live bytes (requested), live objects, fresh/reused addresses, lifetime IDs available (`urevavail`), the heap's own counters: allocations, frees, split, mrev, delin, revoke, init |
| `MQ-HIST` | every 256 samples and at the end | reuse distance: allocations between the free of an address and its next allocation, log2 buckets |
| `MQ-LIFE` | same | object lifetime: allocations between an object's allocation and its free, log2 buckets |
| `MQ-SIZES`, `MQ-STRIDE` | same | requested sizes; distance between consecutive allocations, log2 buckets |
| `MQ-DONE` | end | objects, peak live bytes and objects, peak if rounded to powers of two of at least 256 B, the heap's own totals, objects never freed, the pthread bridge's own allocations set aside (`runtime_allocs`, `runtime_frees`) |

The tracer counts the program's allocations only. The virtual pthread bridge allocates five
records per thread (one of them a 1 MiB exchange buffer) inside `__clone` and frees them at
thread exit; `--wrap` would see those calls, while CheriBSD's tracer cannot see libthr's and
native musl makes none. Allocations inside `__clone`, `__capstone_delegate_thread_attach` and
`__capstone_signals_thread_detach` are therefore set aside with their later frees. They stay in
RSS and pinned pages, as the platform's cost.

sh6bench hands out more distinct addresses than the tracer's table holds, so its reuse and
lifetime histograms follow 1 address in 16 (`MQ_ADDR_SAMPLE=16`), each with
all its reuses, as on CheriBSD.

The native reference (`native/`) uses the same tracer source built with `-DMQ_NATIVE`: reuse,
lifetime, size and stride histograms and the allocation counts, no live bytes.

The CheriBSD side records, per sample, jemalloc's ledger (allocated, active, resident,
mapped, metadata), the quarantine epochs and max RSS; at exit the kernel's sweep counters
(`mqstat.so`); and every `MAPS_EVERY` seconds the resident pages per mapping owner, which
separates MRS's own bookkeeping (`run-maps.sh`). See that experiment's README.

## Checks before a result is used

- **Positive control** (`tracer-check.c`, run first by `gate.sh`): 2 allocations in the main
  thread, 5 in each of 3 threads, all freed, all observable. Passed on 2026-10-09: native
  `objects=17 unfreed=0`; Capstone `objects=17 unfreed=0 runtime_allocs=15 runtime_frees=15`,
  `heap_allocations=34` (17 + 15 + 2 made before the tracer starts), same printed sum.
- **Allocation totals**: a traced run's `ops` must equal the native run's. The programs are
  built hosted with builtins on all three systems, so the compiler may remove an allocation
  whose memory is never used; equal totals show it removed the same ones.
- **Output**: `MB_OUT_SHA256` must equal the hash of the native run's output, except for
  programs that print times (sh6bench), whose output is compared by eye.
- **Completeness**: exit 0, an `MQ-DONE` or `CAPSTONE_VM_STATS` line, `MB_OUT_LINES` equal to
  the streamed line count. A run that fails any of these is not plotted.

## Run matrix

| System | Builds | Runs per program |
|---|---|---|
| Capstone | plain, traced | plain three times (footprint), traced once |
| Native | plain, traced | once each |
| CheriBSD | plain, traced | `on` plain three times; `on` traced; `off` traced; `on` maps |

## Running

```bash
# host: SDK (runtime with this branch's limits), adapter, programs
bash capstone/ports/mallocbench/build-domain.sh <sdk> <mimalloc-bench>/bench <out>
# kit: stage-base/ = <out>/*.dom, mb-run, inputs, capstone_vm.ko, capstone-vexec, gate.sh;
#      qemu-system-riscv64, images/, run-staged.py, remote-probe.sh, SHA256SUMS, PROVENANCE
bash remote-probe.sh <run id> <seconds> <name>...   # one QEMU per name, in parallel
```

`remote-probe.sh` writes `provenance.txt` (stage checksums, QEMU and images) into each run
directory. The native side: `native/build-native.sh <bench> <out>`, then
`native/run-native.sh <out> <results>`.
