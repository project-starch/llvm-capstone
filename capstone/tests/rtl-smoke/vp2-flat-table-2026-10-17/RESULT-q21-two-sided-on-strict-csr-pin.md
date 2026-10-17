# Q-21 two-sided on the pin that makes `0xcc0`/`0x5c0` illegal

The record the board's boot-3-T1 re-run cites. Both arms on **one pinned QEMU binary**, default
properties — nothing sets `x-capstone-test-csrs`.

## The control fires, with the board's exact line

Pre-fix images, five application steps, **five faults**, every one `cause=2 address=0xcc002573`:

| image | sha256/16 (from the fault line itself) | pc | step |
|---|---|---|---|
| contract.dom | `f2017064bea77dd2` | `0x3f94710ccc` | ok |
| contract.dom | `f2017064bea77dd2` | `0x3f94d90ccc` | stale |
| sqlite3.dom | `233640f7193b034c` | `0x3f97211e6c` | sqlite_write |
| sqlite3.dom | `233640f7193b034c` | `0x3fa1411e6c` | sqlite_read |
| mruby.dom | `bba891bc16576d98` | `0x3f83c11a7c` | — |

`R_EXIT` 139 on all five, `VIRTUAL_APPLICATION_OK` absent, no SQLITE/MRUBY values.

**`address=0xcc002573` is the instruction's own encoding** — CSR field `0xcc0`, funct3 2, rd `a0`,
i.e. `csrrs a0, 0xcc0, zero` — which is the board lane's boot-3-T1 line verbatim.

**And the pcs corroborate the static census independently.** The census put the `malloc` read at
`+0x20ccc` (contract), `+0x21e6c` (sqlite3), `+0x21a7c` (mruby); the fault pcs end `...0ccc`,
`...1e6c`, `...1a7c`. Static offsets and dynamic pcs agree without being derived from each other.

Each fault line also carries its image's own sha256, so this record cites images by hash in the
serial rather than by a label beside it.

## The fixed arm passes

```
R_EXIT:ok:0   stale:139   sqlite_write:0   sqlite_read:0   mruby:0   cleanup:0
VIRTUAL_APPLICATION_OK present
SQLITE_SUM=5050   SQLITE_PERSIST=100   MRUBY_SUM=5050
cause=2: 0        address=0xcc002573: 0
```

Exactly one fault in the whole run: `cause=25` on contract.dom `125705371837ea92` at the `stale`
step — the deliberate stale-access fault, with its entry and code range resolved. That is the
intended reading and nothing else.

Images: contract.dom `125705371837ea92`, sqlite3.dom `a5942a1add489597`, mruby `a902e740fc5d3b0f`.
Service steps on this arm: 548 / 782 / 458 / 1748.

  serials `<csr-pair>/{prefix,fixed}/guest/serial.log`, status `<csr-pair>/status`

## Two corrections to records already sent

**1. The pin label on the 03:29 re-records was wrong.** `status-t1` and `status-t2` in the v9 kit say
`pin 852c5bc76774`. They ran at 03:29 against `qemu-v7-wt`, whose **binary had been rebuilt in place
at 03:26** to `3b8040d5ff6e` by another lane. The label came from `git rev-parse HEAD` on a worktree
that had moved — a label drifting from the artefact it names. The readings are unaffected, and in
fact stronger than claimed: those images passed on the pin that makes the CSRs illegal, not merely
on the permissive one.

**Habit this earns:** a QEMU binary under another lane's worktree is not a fixed input. Copy it,
hash it, and probe it BEHAVIOURALLY — this binary was confirmed as `3b8040d5ff6e` by accepting
`x-capstone-test-csrs`, a property that exists only at that revision. `git rev-parse` on the
worktree is a label; the hash and the probe are the identity.

**2. The `+1,033` service-step figure compared arms on two different binaries.** The pre-fix arm
(1,953) ran on `852c5bc76774` and the post-fix arm (2,986) on `3b8040d5ff6e`. The diff between those
revisions is `cpu.c`, `cpu_cfg.h`, `csr.c` and tests — nothing in the service-call path — so the
number should stand, and the direction is certain by construction (three `ensure_ids(256)` sites
that could previously return without a call). But it was presented as a one-binary comparison and
it was not. A strictly matched pair is not obtainable on the default properties, because the pre-fix
images cannot run where the CSRs are illegal; it would need `x-capstone-test-csrs`, which is a
measurement-only use and no kit may set it.
