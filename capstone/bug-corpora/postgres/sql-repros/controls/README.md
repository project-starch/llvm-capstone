# Parent, sibling and successor controls at the memory contexts

Not cases. The PostgreSQL row of the pre-registration
`capstone/docs/plans/2026-10-11-nested-lifetime-controls.md`: does memory-contexts patch 0003
(`PGSU_NESTED=sublet`) leave the accesses it should permit alone, and does it reject the one it
should? They run **inside the backend**, the way this corpus's `pgcorpus_reach` reaches C code: an
extension linked into the image (`module/`, `pgcorpus_lifetime`) and called from SQL in
`postgres --single`, one session per control on a fresh copy of the fixture cluster.

Each control creates a dedicated **AllocSet** -- the context type patch 0003 protects --
with sizes `(0, 8192, 8192)`, which match neither default size set, so aset.c never recycles it
from its context freelists. Every request is 64 bytes (one size class, `fidx` 2). The release
operation is `pfree`, where the patch revokes (`SubletRelease`).

| control | what it does | protected | release |
|---|---|---|---|
| `parent` | palloc D and A, pfree D then A; through the AllocBlock on the context's `blocks` list (the keeper block, as aset.c reaches it), load A's former free-list link and header word, then store into and load back A's former bytes 16..63 (past the link); then one palloc | completes; link = D's chunk, header unchanged, next palloc returns A | same |
| `sibling` | palloc A and B from one block, pfree A, load B (contents intact), store into B, load back | completes, B intact | same |
| `successor` | palloc A, pfree A, palloc until the context returns A's address (C), require equality, store and load through C | completes | same |
| `successor-stale` | the same, then ONE load through the old A | **cause 25** at `lc_probe_stale_read` | completes and reads C's data |

Every access goes through an `lc_probe_*` function in `module/pgcorpus_lifetime.c`, so a fault is
attributed by the function its pc lies in. The module prints its addresses, bounds and checks with
`write(2)` before the access they describe (the stand-alone backend's own output is buffered and is
lost when a fault ends the process). aset.c's private structs are mirrored for their leading fields
only and checked at run time: the block must name the context as its owner and contain the chunk,
and after the pfree the context's `freelist[2]` must be A's chunk.

## Files, and what is copied

- `module/` -- the extension (`lifetime_control(name text)`).
- `build-domain.sh` -- a **copy** of `ports/postgres/app/build-domain.sh` at 53325bba with three
  marked changes: its `SCRIPT_DIR` names the port directory, and it copies `module/` in as
  `contrib/pgcorpus_lifetime` and adds it to `MODULES`. The port's script is unchanged.
- `build-image.sh` -- the postgres branch of `ports/common/application/build-virtual.sh`, calling
  the copy above; same `build.py` arguments.
- `run-controls.py` -- one boot per arm on the virtual kit through `tools/virtualvm.Batch`, the
  same staging and session command as `../shared/run-arm.py` (not changed). Before the controls it
  runs a trivial SELECT and the corpus's own arm control `corpus_control('uaf-palloc')`, which
  must fault on the protected image and complete on the release image.

```
bash build-image.sh OUT_RELEASE
PGSU_NESTED=sublet bash build-image.sh OUT_PROTECTED
python3 run-controls.py --arm protected --virtual-kit KIT --image OUT_PROTECTED/image/postgres.dom \
  --fixture CLUSTER --pg-share SHARE --llvm-bin TOOLCHAIN/bin --raw RAW --out protected.json --repeat 3
```

Results: [`../results/2026-10-11-controls`](../results/2026-10-11-controls/README.md).
