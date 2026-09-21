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

* **`native-fix-differential`** is an extra axis beside protection: natively
  the pair differs by whether the upstream fix is applied, and the driver
  prints what the case observed. In a domain the pair differs by protection,
  and the program prints nothing.
* **`native-detect`** is not merely unwritten: the nodes never reach `malloc`,
  so ASan has no event. Valgrind, against APR's own annotations, is the arm that
  could discriminate.
* **`poisoncap-*`** are declared and not written: no CheriBSD or PoisonCap
  build of APR exists.

## Two targets, one sequence

Real: `apr_pools.c` from the 1.7.4 pin, unmodified but for the two patches the
port [`ports/apr/pools`](../../../ports/apr/pools/README.md) applies — one
replaces fourteen APR includes with the census's shim, the other connects the
node transitions to the adapter. Reduced: the consumer. Each `case.c` writes
its sequence inside `APR_CASE(NN)` and is a complete translation unit on both
targets; [`shared/corpus.h`](shared/corpus.h) is the seam.

    shared/build-cases.sh native <out>            then runners/run-native.sh
    shared/build-cases.sh capstone-domain <out>   then runners/capstone-domain/

The [domain runner's manual](runners/capstone-domain/README.md) has the
commands, the two modes and the oracle. In short: `spatial` must complete,
`sublet` must fault at the labelled probe, the expected address is published by
the run and never hardcoded, and `--negative-control` must make every oracle
say FAIL before a PASS is believed.

## Where this corpus deviates from the contract, and why

`tests/check-corpus.py` in the pymalloc corpus enforces
[SCHEMA.md](../../cpython/pymalloc-repros/SCHEMA.md). Run against these cases it
reports two kinds of problem, both deliberate. They are listed here rather than
silenced, and no copy of that checker is shipped beside them: a fork would be a
second contract, and a checker that fails by design is noise.

| what it reports | why |
|---|---|
| `arm 'native-fix-differential' is not in SCHEMA.md` | the contract's arms differ by **protection**, the defect present in both. This pair differs by whether the **upstream fix** is applied. Folding it into `spatial`/`sublet` would misname it |
| `case.c declares no PYC_CASE` | the macro is the corpus's seam to its allocator; here it is `APR_CASE`. The rule the checker means — a case declares the number its directory carries, and the driver refuses a fixture that names another — is implemented |

Extending the checker to know these is a change to the pymalloc corpus and
belongs in a conversation with it, not a unilateral edit from here.
