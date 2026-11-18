# Sublet lifetimes for gcc's own allocators (SPEC CPU2017 502.gcc_r)

The kit's build of 502.gcc_r protects what `malloc` returns. gcc carves its own objects
out of that memory with three allocators, and a reference kept past `pool_free`, `ggc_free`
or an `obstack_free` still reads and writes the released object. This adapter makes every
such object a lifetime of its own. It uses the one-operand `CREVOKE` and `CRENEW`
([isa.md](../../../../docs/design/virtual-capstone/isa.md)), so it runs on QEMU. The RTL has
the two-operand `CREVOKE` and no `CRENEW`, and cannot run it.

| Allocator | Hand-out | Release | Kept by the allocator |
|---|---|---|---|
| Pools (`alloc-pool.c`) | `CDERIVE` when an element leaves its block for the first time. A reused element costs nothing | `pool_free` renews the element (`CRENEW`) and puts the renewed reference on the free list | nothing; the element layout is stock |
| GC heap (`ggc-page.c`) | `CDERIVE` in `ggc_alloc`, bounded to the size-class slot | `ggc_free` ends the object (`CREVOKE`) | nothing |
| Obstacks (`obstack.c`, `obstack.h`) | `CDERIVE` in `obstack_finish` | a free to an address ends each object it cuts off (`CREVOKE`) | the reference of each live object, 16 bytes |

A block, a chunk and a page group are `malloc` objects. Freeing one ends every object carved
from it, with no code in the port.

No release looks anything up. `CREVOKE` and `CRENEW` take the client's reference, and every
hand-out site already holds its parent: the pool's block cursor, the page, the obstack's
object base.

**Why the obstack keeps references.** `obstack_free (h, obj)` frees `obj` and everything
allocated after it, and gcc frees to marks: an object allocated once and freed to many times.
After the first free the mark's reference is dead and only its address is good. So the
obstack frees by address, and ends each object that a free cuts off through the reference it
kept. Freeing a whole chunk needs none of them.

The port is 158 added and 8 removed lines in five files, counted as a diff of the ported
tree against the stock one: `alloc-pool.c` 7, `ggc-page.c` 8, `obstack.c` 90,
`include/obstack.h` 15 and the new `sublet-port.h` 38. The fix below adds 16 lines to
`df-scan.c`.

## One fix in gcc itself

With the pools ported, gcc stops on its own test input: `forward_propagate_into` loads
through a df ref's `insn_info` whose lifetime has ended (cause 25). `df_insn_rescan` collects
an insn's new refs with the insn's current record as their `insn_info`, then frees that
record to its pool, allocates a record again and installs the collected refs. They still
name the freed record. Stock gcc works because the pool's free list hands the same element
out again at once. `apply.py` makes `df_insn_rescan` keep the record, as GCC does from
release 6.1 on. No lifetime instruction is skipped and no pool is exempt.

## Reproduce

The licensed sources and every build stay outside the repository.

```sh
source capstone/tests/capstone-test-env.sh
K=capstone/ports/spec-cpu2017
A=$K/sublet/502.gcc_r

# The stock capability build of the benchmark (the kit's README), with an SDK whose heap
# lifetimes are on. LABEL is a name of your choice.
BENCHMARKS=502.gcc_r bash $K/tools/build.sh $K $W/tree/SPEC17 $SDK $W/build LABEL
STOCK=$W/tree/SPEC17/benchspec/CPU/502.gcc_r/build/build_base_LABEL.0000

# The ported tree: a copy of the build directory, the port, the stock build's commands again.
cp -a $STOCK $W/ported
python3 $A/apply.py $W/ported --record $W/source-audit.json
bash $A/replay-build.sh $W/ported

# The ported allocators begin and end lifetimes, the stock ones do not.
python3 capstone/ports/common/application/lifetime-instructions.py $W/ported/alloc-pool.o --expect present
python3 capstone/ports/common/application/lifetime-instructions.py $STOCK/alloc-pool.o --expect absent

# Both executables on SPEC's test input, in one guest boot.
T=$W/tree/SPEC17/benchspec/CPU/502.gcc_r/data/test
REFERENCE=$T/output/t1.opts-O3_-finline-limit_50000.s ROUNDS=9 \
  bash $A/run-test.sh $PLATFORM $T/input/t1.c $W/run stock=$STOCK/cpugcc_r sublet=$W/ported/cpugcc_r
```

- `apply.py` refuses a tree whose files differ from `baseline.json`, and fails unless the
  ported files have the hashes pinned there.
- `replay-build.sh` runs the compile and link commands of the stock build's `make.out` again,
  so compiler, flags and SDK are the stock build's. On an unchanged copy it must give the
  stock executable byte for byte.
- `run-test.sh` drives `capstone/runtime/virtual/run-staged.py` of the checkout whose
  environment is sourced. That checkout must match the platform kit.
- `icount.c` is the guest helper for `ICOUNT=`, which adds the instructions of each job.

## What is verified

[results/2026-11-18-test](results/2026-11-18-test/README.md): on SPEC's test input the
ported compiler gives SPEC's reference output in every run, the job runs 0.58 % more
instructions, and the control without the gcc fix faults. The runs used the kit of revision
`71ce199d8` of this branch (native task route).

## Limits

- **Test input only.** `t1.c` is 69 bytes. No larger input has run with this port, so nothing
  here says how often gcc touches a released object on real compilations, or what the port
  costs there.
- **Not run on the trap route** of this branch's head, and not on the board.
- **Objects of 4 KiB or more keep their block's lifetime.** `CDERIVE` refuses bounds it
  cannot hold exactly, and the port does not pad. This affects GC objects of 4 KiB or more
  and obstack objects of 4 KiB or more.
- **A zero-size obstack object has no lifetime.** `CDERIVE` needs a positive size. Such an
  object is a mark, and the client gets the obstack's own reference at that address.
- **A growing obstack object is not a lifetime until `obstack_finish`.** `obstack_base`
  hands out the obstack's own reference, as stock.
- **No collector.** SPEC compiles `ggc_collect` to an immediate return, so a GC object ends
  only at `ggc_free`. A build that sweeps would need the heap to keep the references it
  issued, as the obstack does.
- **The typed free lists on top are not ported**: bitmap elements, `lists.c`, and the others
  that keep released objects of one type for reuse.
- **The obstack's references cost memory**: 16 bytes for each live obstack object.
