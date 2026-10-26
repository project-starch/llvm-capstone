# Slot census: how many heap objects are live at once

How many lifetime-id slots does a process need? The decision rests on one number per program, input and dataset
size: the exact whole-run peak of simultaneously live heap objects in any one process. This directory measures it
natively on x86 with an `LD_PRELOAD` tracer, starting with SPEC CPU2017. Capstone is not involved, so the program's
own allocation behaviour is measured, not a port's.

    slotcensus.c     the tracer: an exact live counter at every malloc/calloc/realloc/free/memalign event (C++
                     new/delete reach it through libstdc++), the running peak, allocation and free counts, realloc
                     moves and in-place resizes, peak live usable bytes, a log2 histogram of requested sizes; one
                     JSON line per process at exit. A fork child starts at the live set it inherits.
    control.c        positive control: a known event sequence (peak, every entry point, both realloc classes,
                     realloc(NULL)/realloc(p, 0), a fork child) that prints the counts it must produce
    control_cxx.cc   positive control for operator new[]/delete[]
    check-control.py compares a control's EXPECT lines with its census lines; no census line is a FAIL
    run-spec.sh      SPEC CPU2017 rate, C and C++ (16 benchmarks; no Fortran compiler on the measuring host)
    summarize.py     per-invocation and per-benchmark peaks plus runcpu's validation verdicts
    pack.py          copies a pass into results/<stamp>/ with the host's paths removed

## Running

    bash run-spec.sh ~/spec2017/SPEC17 <pass-dir> [test|train|ref] [benchmark...]
    python3 summarize.py <pass-dir> [summary.json]

`run-spec.sh` builds the tracer and the controls, then runs three controls: C (with its fork child) and C++ under
the tracer must pass, and C without the tracer (the negative control) must fail. Any other outcome exits 75 before
SPEC is touched. It then writes `config/slotcensus.cfg` from SPEC's `Example-gcc-linux-x86.cfg` (label `census`,
the system GCC, `-O2`, `-fcommon -fpermissive`, `-std=gnu++14` for C++, 502.gcc_r's
`-fgnu89-inline -D_FORTIFY_SOURCE=0`, 526.blender_r's `-std=gnu99`, without which glibc hides `struct timespec` from
its `threads.c`), builds every benchmark once, and runs each benchmark in its own `runcpu`, all
in parallel; the counts do not depend on timing. runcpu validates every output with SPEC's own tools, and
`summarize.py` refuses a benchmark without a `valid` verdict or without census lines.

The control is built with `-fno-builtin`: at `-O2` GCC deletes a `malloc` whose result is unused and a
`calloc`/`aligned_alloc`/`realloc(NULL)` that is freed at once. The first version of the control lost 3025 events
that way and its check failed, which is how this was found. The benchmarks are built normally, so an allocation the
compiler removes is not an object and is rightly not counted.

What the tracer cannot see: objects inside a program's own allocator (a pool or arena carved from one malloc'd
block counts as one object), allocations made by `dlsym` while the tracer resolves the real functions (a 64 KiB
static buffer, not counted), and processes that end in `_exit` or a signal (no line).

## Results

`results/<stamp>/` holds a pass's raw lines with host paths removed (`pack.py`): the control files, each benchmark's
census lines (benchmark processes only: not the shells, specinvoke, specdiff or SPEC's image validators), the
generated config with the pass directory replaced by `<PASS>`, the provenance record, and runcpu's
`Success:`/`Error:` lines. runcpu's full logs carry the host's paths and stay on the host.
