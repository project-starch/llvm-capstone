# PostgreSQL 17.5: protected build gives the same answers, 2026-11-11

`work.sql` in the single-user backend again, after the lifetime instructions changed (`CREVOKE`
takes the object alone, `CRENEW` exists) and memory-contexts patch 0003 was converted to them.
The comparison is the one of [`../2026-10-11-same-answers`](../2026-10-11-same-answers/README.md):
the protected build must give the 22 result rows and the final count of the stock build and of
native PostgreSQL 17.5. `result.json` is the record.

**Outcome: it holds.** All three images exit 0 without a domain fault, print 22 rows ending in
`count = "1500"`, and every row's sha256 equals the row hash the 2026-10-11 record holds for the
rows that matched native 17.5. No native run was made in this session; the oracle is that record.

| arm | image sha256 | exit | domain faults | rows | equal to the recorded native rows |
|---|---|---|---|---|---|
| stock (`PGSU_NESTED=none`) | 9f3f6c87… | 0 | 0 | 22 | yes |
| protected, patch 0003 as converted (`c63524ea…`) | e7efdd7d… | 0 | 0 | 22 | yes |
| protected, patch 0003 of this tree (`29b94c5a…`) | 260e63bd… | 0 | 0 | 22 | yes |

The launcher's per-process statistics (`CAPSTONE_VM_STATS=1`):

| arm | nodes | collections | reclaimed | node capacity |
|---|---:|---:|---:|---:|
| stock | 4461 | 0 | 0 | 5120 |
| protected, as converted | 62066 | 5 | 57377 | 17408 |
| protected, this tree | 62079 | 5 | 57371 | 17408 |

The 2026-10-11 record gives `nodes=65233 collections=5 reclaimed=53787` for its protected arm and
`nodes=6203` for stock. That run used the earlier patch, runtime and emulator, so the figures are
not a one-variable comparison. The two protected images here differ by the block's own
pointer in the block header, which the port's result record explains
(`../../../memory-contexts/results/2026-11-11-virtual`); on this workload it changes neither the
answers nor, visibly, the node counts.

## What ran

- **Builds.** `ports/common/application/build-virtual.sh postgres` with
  `CAPSTONE_VIRTUAL_LIFETIMES=0` and `=1` (the entry refuses `PGSU_NESTED=sublet` on its own, as a
  contradiction of the selector's default), compiler sha256 `0bbb1f88bc90…`, the pinned archive
  `fcb7ab38…`. The tree was branch `one-operand-crevoke-crenew` at 59201bdf2 as a plain copy
  without `.git`, so the manifests record `runtime_dirty: true`.
- **One boot**, each arm on its own `cp -a` of the kit's pristine cluster (tree `0353a5b9…`), as
  `nobody`, with the `--single` arguments of the 2026-10-11 gate script and
  `-c exit_on_error=true`.
- **Platform.** The virtual kit built from this tree: QEMU `2d5a7107890e…`, launcher
  `0f8e0f903725…`, module `e47bca62addf…`, pinned mode with the module's defaults, `-cpu` extras
  `x-capstone-test-csrs=true,x-capstone-exact-bounds=true`.
- **N.** 1 per arm.
