# httpd / APR pool defect corpus

Consumer-side defects in code that allocates from `apr_pool_t`. A destroyed or
cleared pool's nodes do not go back to `malloc`: `allocator_free`
(`apr_pools.c:414`) pushes them onto a size-bucketed LIFO free list, and only
nodes past `max_free_index` are ever released — under the default
`APR_ALLOCATOR_MAX_FREE_UNLIMITED`, none are. The next `apr_pool_create` pops
the same node back.

    00_9e6be73065_watchdog_destroyed_pool_reused/   the handle outlives the pool

| shape | cases |
|---|---|
| stale allocator handle / reuse / allocation through the dead handle | 0 |

One case, one shape.

## The contract

The layout and the `case.json` fields are the corpus contract in
[`cpython/pymalloc-repros/SCHEMA.md`](../../cpython/pymalloc-repros/SCHEMA.md),
referenced rather than copied. Where this corpus differs:

* **`native-fix-differential`** replaces the protection axis: this pair differs
  by whether the upstream fix is applied.
* **`spatial` and `sublet` are declared and not written.** The APR port is a
  compilation census, not a domain workload, so no protected arm exists yet and
  the gap is visible in each `case.json` rather than silently absent.
* **`native-detect`** is not merely unwritten: the nodes never reach `malloc`,
  so ASan has no event. Valgrind, against APR's own annotations, is the arm that
  could discriminate.

## Running

    bash runners/run-native.sh [outdir]

Real: `apr_pools.c` from the 1.7.4 pin, unmodified, with `apr_shim.h` standing
in for the fourteen headers a configure run would have generated. Reduced: the
consumer. `shared/stubs.c` answers the symbols the allocator references and no
case reaches — and each one **aborts with exit 75** rather than returning
quietly, so a case that came to depend on a stub could not pass unnoticed.

## Where this corpus deviates from the contract, and why

`tests/check-corpus.py` in the pymalloc corpus enforces
[SCHEMA.md](../../cpython/pymalloc-repros/SCHEMA.md). Run against these cases it
reports exactly two kinds of problem, all of them deliberate. They are listed
here rather than silenced, and no copy of that checker is shipped beside them: a
fork would be a second contract, and a checker that fails by design is noise.

| what it reports | why |
|---|---|
| `arm 'native-fix-differential' is not in SCHEMA.md` | the contract's arms differ by **protection**, the defect present in both. This pair differs by whether the **upstream fix** is applied. Folding it into `spatial`/`sublet` would misname it |
| `case.c declares no PYC_CASE` | the macro is the corpus's seam to its allocator; here it is `APR_CASE`. The rule the checker means — a case declares the number its directory carries, and the driver refuses a fixture that names another — is implemented |

Extending the checker to know these is a change to the pymalloc corpus and
belongs in a conversation with it, not a unilateral edit from here.
