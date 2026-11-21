# A liveness-test instruction: per-benchmark census (2026-11-21)

The runtime lane asked for this census in the stage-2 shape. **Result: well under the 0.5 % gate on every row.** The
highest is perl, at 0.032 % of instructions and 0.16 % of loads.

## What a liveness test would replace

Today a liveness test is a load probe. `heap_access` (`ports/musl-capstone/arch-capstone64/heap_sublet.h`) executes
`lbu zero, 0(p)`, so a revoked or stale capability traps before the allocator recovers mapping authority. A
dedicated test would replace that load with a check that touches no memory:
- **instructions:** one replaces one, so none is removed;
- **data accesses:** one load is removed per probe;
- **semantics:** the probe also requires read permission and a non-empty bound on `p`; a test would not.

## Instrument

- Arm CRVC0, shift-1 twin profiles over every command window, each benchmark's `.full` image.
- Every executed load into `zero`, attributed to its function.
- The only executed ones are mallocng's probes: `__libc_free`, `__libc_realloc` and `__malloc_allzerop`. Other
  loads into `zero` exist statically (perl's `my_re_op_compile`) and execute 0 times.

## Readings

| row | probe executions | % of instructions | % of loads |
|---|---:|---:|---:|
| perl test-sw | 26,682,900 (free 18.2 M, allzerop 5.8 M, realloc 2.7 M) | 0.032 % | 0.162 % |
| gcc train01 | 1,228,000 | 0.010 % | 0.042 % |
| mcf test | 119,878 | 0.000 % | 0.001 % |
| nab test | 5,002 | 0.000 % | 0.001 % |
| x264 test | 3,437 | 0.000 % | 0.000 % |
| imagick test | 740 | 0.001 % | 0.006 % |
| xz test | 444 | 0.000 % | 0.000 % |
| lbm test | 6 | 0.000 % | 0.000 % |

## Reading

- As a performance lever the instruction does not pay: at most 0.16 % of loads, and no instruction removed.
- Its case, if any, is semantic: a test that needs no read permission and no bytes in bounds, so a write-only or
  zero-length capability could be checked. That is an ISA question, not a measurement.
- These counts are for the heap with lifetimes. A heap that probes on every access instead of only on free and
  realloc would change them.
