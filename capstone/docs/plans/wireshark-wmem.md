# Wireshark wmem on Capstone + Sublet

*Written before the first QEMU run, so the predicted readings below bind the
go/no-go and are not fitted afterwards. Candidate selection and the defect
inventory are in `docs/ref/port-candidate-survey.md`.*

*Measured 2026-09-21: every predicted reading held — 26 of 26 paired verdicts,
both replay modes complete and identical. Evidence in
`ports/wireshark/wmem/results/20260921-qemu/`. Two infrastructure attempts
preceded it, both retained in the raw archive: a runner host without
`pexpect`, and a replay image of 2,901,776 loadable bytes that the loader
module refused where 1,330,288 had loaded — an image without a
`.capstone_domreq` declaration is sized by the module's default, so the
replay's static tables were reduced. One prediction was corrected before the
passing run: the recycler writes its free-list node into the freed chunk, so
the unprotected reads of fixtures 3 and 4 return allocator metadata rather
than the old byte.*

## Goal

Port Wireshark's memory manager `wmem` — its core and all four allocators —
to the Capstone domain with the two protection modes every other component
port has: `spatial` (every object narrowed to its request) and `sublet`
(additionally, a pool reset ends the epoch of every object in its retained
blocks). Pin **wireshark 4.6.8**; both block allocators are blob-identical
from 4.4.0 through 4.6.8, so nothing here depends on the exact release.

## What is ported

| Unit | Role in Wireshark | On this port |
|---|---|---|
| `wmem_allocator_block_fast.c` | the per-dissection packet pool; 2 MiB blocks, bump, `free` is a no-op, reset keeps the first block | reset renews the retained block's authority |
| `wmem_allocator_block.c` | file and epan scopes; 8 MiB blocks, master + recycler free lists, split and coalesce | reset renews every retained block and rebuilds the block list |
| `wmem_allocator_strict.c` | the debug arm under which all reported defects were found | every object is its own region; `free` renews it |
| `wmem_allocator_simple.c` | pass-through wrapper used by a handful of decompressors | every object is its own region |
| `wmem_core.c`, `wmem_user_cb.c` | dispatch, scope flags, callbacks | unchanged; GLib replaced by a shim |

The scope layer (`epan/wmem_scopes.c`, the `epan_dissect_t` pool cache in
`epan/epan.c`) is modelled in `src/shared/scopes.c`, not extracted: the
upstream files pull the whole of `epan/`.

## Predicted readings, per fixture and mode

`security-tests/shared/lifetimes.c`, run by `security-tests/qemu/run.py`.
A fault verdict requires the stage marker, the expected cause, and the exact
PC of the labelled access. `completed` requires status 0.

| # | fixture | spatial | sublet |
|---|---|---|---|
| 0 | live controls; storage reuse after reset asserted for both allocators | completed | completed |
| 1 | packet-pool reset, stale read (the reported shape) | completed, old byte read | fault at read |
| 2 | reset, same storage reissued, stale interior write | completed, new object corrupted | fault at write |
| 3 | recycler `free_all` retains and reinitializes its block, stale read | completed, reads the free-list node | fault at read |
| 4 | recycler individual `free`, stale read — documented limit | completed, reads the free-list node | **completed** |
| 5 | one byte past the request | fault, OOB | fault, OOB |
| 6 | strict allocator `free`, stale read | completed | fault at read |
| 7 | 2000 reset epochs on one retained block, stale read | completed | fault at read |
| 8 | pool destroyed, stale read | completed | fault at read |
| 9 | jumbo object released by reset | completed | fault at read |
| 10 | file scope left (reset + gc returns the block) | completed | fault at read |
| 11 | packet pool's second block returned by reset | completed | fault at read |
| 12 | stale pointer handed back to the allocator | completed (not attempted) | fault at the allocator's probe |

If case 4 faults in sublet mode the port has invented a revocation the
allocator cannot support and the adapter is wrong. If case 0 fails, the
storage-reuse property this port exists to study is not present as assumed.
If any protected case completes instead of faulting, the epoch hook missed a
retained block.

## What the replay proves, and what it does not

`tests/native/test-replay.py` drives all four allocators through allocation,
resize, individual free, reset, collection and destruction, and requires the
hooked build to report byte-identically to an unhooked reference build.
Under QEMU the same trace must complete in both modes. That establishes that
the hooks do not change allocator behaviour for correct programs. It is not
a workload recording from tshark; a recorder is a follow-up.

## Limits stated up front

* An individual `wmem_free` in the recycler allocator is not revoked (case 4).
  Sublet lends whole regions; a chunk inside a live block has none of its own.
* Regions are reissued only at their exact size and never returned to the
  payload. A long replay with many distinct sizes can exhaust it.
* Nothing here runs a dissector. The reported defects are in
  `bug-corpora/wireshark/wmem-repros/`: seventeen reports, thirteen distinct
  defects, twelve revoked reads and one recorded non-detection, measured
  2026-09-21.
