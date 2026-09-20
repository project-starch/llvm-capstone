# httpd / APR pool defect corpus

Consumer-side defects in code that allocates from `apr_pool_t`. A destroyed or
cleared pool's nodes do not go back to `malloc`: `allocator_free`
(`apr_pools.c:414`) pushes them onto a size-bucketed LIFO free list, and only
nodes past `max_free_index` are ever freed — in the default
`APR_ALLOCATOR_MAX_FREE_UNLIMITED` configuration, none are. The next
`apr_pool_create` pops the same node back.

    9e6be73065_watchdog_destroyed_pool_reused/   the handle outlives the pool

`run.sh` builds against upstream's `apr_pools.c` byte for byte, through the seam
the [APR census](../../../ports/apr/census-apr.sh) established, and runs the
control arm first.

## Scope

Real: `apr_pools.c` from the 1.7.4 pin, unmodified, with `apr_shim.h` standing
in for the fourteen headers a configure run would have generated. Reduced: the
consumer. `shared/stubs.c` answers the symbols the allocator references and this
fixture does not reach — and each one **aborts with exit 75** rather than
returning quietly, so a case that came to depend on a stub could not pass
unnoticed.

These are native paired arms. No Capstone or Sublet arm exists: the APR port is
a compilation census, not a domain workload.
