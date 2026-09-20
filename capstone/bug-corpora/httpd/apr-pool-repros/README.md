# httpd / APR pool defect corpus

Consumer-side defects in code that allocates from `apr_pool_t`. A destroyed or
cleared pool's nodes do not go back to `malloc`: `allocator_free`
(`apr_pools.c:414`) pushes them onto a size-bucketed LIFO free list, and only
nodes past `max_free_index` are ever released — under the default
`APR_ALLOCATOR_MAX_FREE_UNLIMITED`, none are. The next `apr_pool_create` pops
the same node back.

    00_9e6be73065_watchdog_destroyed_pool_reused/   the handle outlives the pool

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
