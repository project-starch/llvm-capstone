# SQLite 3.22.0 engine-internal defects, against a nested allocator

43 defects in SQLite's own engine, reduced to one domain program each, and run
against the arena allocator SQLite ships with. The sibling corpus
[`../capi-repros`](../capi-repros) covers pointer-lifetime defects in host
*bindings* over the C API and records engine-internal defects as out of scope;
these are those.

**Why memsys5 is the boundary.** memsys5 hands out `&mem5.zPool[i*szAtom]` — a
pointer derived from one 256 KiB arena. Every sub-allocation therefore inherits
the arena's bounds, so a use-after-free lands inside a live object and an
overflow that stays in the arena is in bounds. A capability machine is correct
to stay silent on both, and ASan is blind for the same reason: the arena is one
static array it never sees carved up. That is the measurement, not a caveat.

## Shapes

Every case declares one of these, and the table partitions the corpus.

| shape | cases | why it is not the same test |
|---|---|---|
| free / stale read | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 | the base case: a lifetime ends and a surviving pointer reads the block |
| read past the end of the record | 28, 30, 31, 32, 33, 34, 35, 38, 39 | the pointer is live; the decode or walk leaves the object |
| stale or wild pointer dereference | 27, 29, 36, 37 | the slot holds a value that was never a valid pointer, so there is no block to protect |
| non-terminating loop | 40, 42 | no memory error at all; the resource exhausted is time |
| negative length into memcpy | 26 | the length is computed, goes negative, and is passed as a size_t |
| unbounded recursion | 41 | the resource exhausted is the call stack, not the heap |

## Arms

| arm | target | what it answers |
|---|---|---|
| `spatial` | Capstone domain, base | does the unprotected machine notice? |
| `sublet` | Capstone domain, nested-allocator discipline | does per-sub-allocation revocation notice? |
| `cheribsd-revocation` | CheriBSD riscv64-purecap, revocation on | does CHERI plus revocation notice? |

## A silent arm only means something with a probe

`results/20261004/matrix.tsv` has 32 `silent` rows on `spatial` and 26 on
`cheribsd-revocation`, and **those two silences are not the same claim.**

The CheriBSD runs carry a defect-site probe: a marker at the exact `sqlite3.c`
line where the host ASan oracle, running the same case source with the same
flags, reported the error. Each oracle string says whether the site was reached
(`hits`) and whether the defective access itself was witnessed. A silent row
there is a usable negative. Three rows are `not-reached`, and before the probe
existed those three were being counted as "the mechanism missed it".

The `spatial` arm has no such probe yet. A marker does not survive there: the
domain's `out_text()` writes a shared region the host reads only after the
domain returns, so a faulting domain prints nothing and the marker is lost with
it. On Capstone the equivalent evidence is a symbolized fault pc. Until that is
done, a silent `spatial` row means the exit code was clean and nothing more.

## Running it

The runner is `capstone/ports/sqlite/repro322/corpus322.sh`, by group:

    corpus322.sh build <group>
    corpus322.sh run   <group>

Groups exist because cases need different `SQLITE_ENABLE_*` flags, and the flag
is sometimes load-bearing: case 10 needs `SQLITE_ENABLE_FTS3_PARENTHESIS` or the
parentheses are ordinary characters, a flat query runs, and the case passes
having tested nothing. Cases 38, 39 and 42 need `SQLITE_ENABLE_DBSTAT_VTAB` or
the statement fails with "no such table: dbstat".

Two options make a run much cheaper, and they only work together:

* `SQLITE_DOMAIN_FAULT_RECOVERY=1` at build time, and
* `CAPSTONE_QEMU_BINARY` pointing at a trap-delivery QEMU
  (`capstone-qemu` branch `runtime/1-domain-trap-delivery`, commit `77d69353b7`).

With both, a capability fault returns through the domain's caller instead of
terminating QEMU, so one boot runs a whole group. Measured: a group of eleven
cases used to need six boots because three of them faulted; the base arm now
runs 43 cases with zero VM halts. Hangs still cost a boot — fault recovery
cannot contain a domain that never returns, which is why cases 40, 41 and 42 are
the expensive ones.

## What is not here

Five further R2 fuzz-diff images (`fz01`, `fz03`, `fz04`, `fz05`, `fz07`) have a
host-ASan signature but no derived trigger: fuzzcheck pairs every database with
every script and the provenance kept only the database id, so the SQL has to be
re-derived by replay. Eight more entries in the collection are instruments
rather than defects — five baseline probes and three allocator demonstrators —
and stay in the port.
