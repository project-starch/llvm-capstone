# Trusted-Linux M1 boundary model

This is an executable, finite model of the transitions proposed in
[the M1 execution boundary](../../docs/design/trusted-linux-execution-boundary.md).
It tests whether object authority and ordinary page translation stay paired
through a context switch, fault retry, retirement and private clone. It is a
pre-implementation counterexample search, not a proof of QEMU, RTL or Linux.

Run from the repository root:

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/models/trusted-linux-m1/model.py \
  --record capstone/models/trusted-linux-m1/result.json
```

The model has two address-space instances and two harts. Both spaces reuse the
same numerical ASID, virtual page, object node number and initial generation;
their physical frames and lifetime records are distinct. A hart has separate
user, lifetime and translation selectors. Its one-entry authorization cache is
indexed by the reusable ASID and generation. Correct switching clears that
cache. Each hart can have one already checked access, which retains a frame and
backing epoch after the page table or object node changes.

`mmap` reserves a one-page VMA and returns abstract tagged arena authority;
the one modeled object is created inside it, with physical backing installed
only after a fault. `issue` checks tag, liveness, generation and PTE
permissions before creating a pending access. `complete` applies a store to
the selected frame. `retire` blocks new authority. Each hart then invalidates
its authorization cache and independently acknowledges a drain; a drain is
refused while that hart has an older access to the retiring context. `finish`
permits reuse only after both harts acknowledge. `reuse` advances both the
object generation and the backing epoch. `unmap` additionally removes the
PTE, but an already issued access still targets its old physical frame. A page
fault records the instruction's arguments; after Linux installs a PTE,
`retry` checks object authority again. The one-buffer `read` transition
validates the *requested* length before consuming input. `clone` copies
private lifetime state and page
contents into a distinct frame; dead identities stay dead.

The retirement break is modeled as globally visible for its context before
`retire` returns. It blocks new accesses to that context while per-hart cache
invalidation is still under way; an unrelated context may issue. This is a
hardware/ABI contract to implement and price, not a measured property of the
current core.

The explorer enumerates every ordering of seven barrier/completion/reuse
events after an issued access and retirement: 5,040 schedules each for `free`
and `unmap`. Nine directed scenarios test selector/ASID reuse, `mmap` and fault
retry, PTE write permission, the syscall span, live/dead private cloning and
an unrelated context issuing during retirement. Rejected actions
must leave the state unchanged. Non-vacuity gates require successful issue,
completion, finish and reuse schedules; a successful child access; a denied
retry and inherited stale pointer; and an unrelated issue during retirement.
Every correct family must be safe. Ten injected variants must each fail at its
intended property; the missing free drain is exercised in two families.

The state space is intentionally tiny: one node and one page per context, two
generations, one pending access per hart, a single authorization cache entry,
and fixed event multisets. Schedule permutations are exhaustive *within those
families*, not over all possible programs or memory states. Page contents are
integers; capability representations, register tags, bounds compression,
multi-page accesses, linear moves, kernel fault recovery, asynchronous I/O,
TLB internals and RTL timing are outside this model. In particular, the
`read` transition covers preflight and input consumption, not concurrent
kernel copy recovery. The next prototype must exercise those omitted paths.
