# Fetch-face discriminator pair: invalidating lifetime ops before escapes, or none

One source, `revoke-loop.c`, built twice with the corpus SDK t3-freeze-nb (`capstone-cc -O1
-I<repo>/capstone/runtime/include`), once plain and once with `-DWITH_REVOKES`. The only difference is the #ifdef
around the inner loop, which per round derives a 64-byte child of a static arena's capability (CDERIVE) and revokes
it (CREVOKE). Each round then spins long enough to take quantum escapes.

    CAPSTONE_VM_STATS=1 ./capstone-vexec revoke-loop[-ops].dom ROUNDS INNER SPIN     (defaults 200 64 2000000)

| image | sha256 |
|---|---|
| revoke-loop.dom | 252c16ca3c63cec538510ba532fd48a660670560a29005693378a97d6429c2be |
| revoke-loop-ops.dom | cf8be80bec87bc039bb1ccd0fc73df2183d3c9a3faa0fb71d59e888a0600dc9f |

**Census by raw word, per function** (`sites.py`, FUNC symbols; 28,618 and 28,649 instructions parsed):

| | `main` | the linked runtime (malloc, `__capstone_malloc_prepare_free`, `__capstone_malloc_nested_release`) |
|---|---|---|
| plain | CREVOKE 0, CDERIVE 2 | CREVOKE 3, identical in both |
| -ops | CREVOKE 1, CDERIVE 4 | CREVOKE 3, identical in both |

An image-wide count cannot read 0, because every image links the runtime's three CREVOKE sites (dyn-census.py: 3
against 4). `main` calls neither malloc nor free, and stdio here buffers statically. Whether startup reaches malloc's
site is NOT certified; if it does, it does so identically in both arms, once, before the first round.

**QEMU smoke** (kit qkit7, the images' launcher ABI), default arguments:

| image | FF_DONE | CAPSTONE_VM_STATS |
|---|---|---|
| plain | revoke_pairs=0 | steps=12709 faults=16 nodes=23 |
| -ops | revoke_pairs=12800 | steps=12695 faults=16 nodes=12823 |

Both exit 0. The node count shows the 12,800 derive/revoke pairs executed. The vp2 flat module never collects, so
`ROUNDS*INNER` plus about 23 startup nodes must stay below the table's capacity (node_flat_records, 65,536 at the
default page count), or the -ops arm ends in cause 30 like dense-churn.
