# Services inside CV_STEP

NODES is the only service consumed inside `CV_STEP`. It uses the same
quota validation, refill and collection helper as `CV_NODES`, under
`vm_lock` with interrupts and preemption enabled. `internal_nodes=0`
retains the launcher route. Other service IDs remain external.

A completed internal request owns a pending scalar reply in its thread.
If a signal, scheduling request or deadline stops execution before the
next entry, the launcher sees a quantum event while the saved service
event and reply remain intact. The next step requires no launcher reply
and resumes with action 2. Only an actual entry consumes the pending
reply. An already-expired budget leaves the request to the launcher.
This also prevents replay or replacement of an outstanding capability
reply. The `internal_nodes` statistics field requires a matching launcher
and module.

`run-node-services.py` checks successful and refused requests, launcher
fallback, stops after handling, initial-entry retries, SIGKILL, empty
registries and balanced pins after flushing per-CPU VM counters. These
are NODES fault/stop tests; they do not qualify a mapping transaction.

## Why MAP and UNMAP remain external

The supplied Linux 6.1 kernel exports `vm_mmap` and `vm_munmap`, but that
pair does not provide the required transaction around irreversible grant
retirement. In `mm/mmap.c`, `do_mas_align_munmap` can return `-ENOMEM`
from `mas_preallocate` before deciding whether to split a VMA. Even
unmapping a whole VMA is therefore fallible. `__vm_munmap` also takes
`mmap_write_lock_killable` and can return `-EINTR`.

Retiring first and then returning a failed unmap leaves a mapping without
its grant. Unmapping first violates retire-before-release. Re-minting a
grant after failure cannot restore the identities of revoked descendants.
Similarly, mapping first and then failing to mint requires rollback via
the same fallible unmap. Retrying indefinitely or keeping an ungranted
tombstone does not establish exact recoverable rollback. The inherited
launcher sequence has this failure gap; copying it into the module would
not resolve it.

No in-module MAP/UNMAP path is included. A future implementation needs a
proved preparation/commit interface with no fallible release after
retirement, or a separately specified failure contract, plus injected
allocation and partial-failure tests that check both mapping and grant
state. This is a limitation of the proposed implementation with the
available APIs, not a proof that every possible design is impossible.

The launcher's AVL mapping registry also remains authoritative for its
existing validation. Moving mapping services requires moving visible
length, mapping kind and transport holds into module arena records and
providing trusted query/hold operations for protection, remap, futex,
attachment and thread-exit checks. Returning an internally created map
without updating those consumers is insufficient.

PROTECT stays external because the supplied export table has no
`do_mprotect_pkey` or equivalent module entry. No Linux or ISA changes
are part of this milestone.
