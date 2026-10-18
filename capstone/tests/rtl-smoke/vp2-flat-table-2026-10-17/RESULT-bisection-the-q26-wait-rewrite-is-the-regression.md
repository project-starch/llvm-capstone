# Result: the bisection names the Q-26 wait rewrite, and it is reverted

A matched pair, both arms booted, both as pre-registered.

| variant | banner tag | reading |
|---|---|---|
| **q26only** — the Q-26 wait rewrite alone | `bisect-q26only` | `vm-contract` retval 922; both contexts `cause=24 pc=…1d94 address=0x2 entry=unknown` — the `memset+0x2c` destination-of-2 signature |
| **esconly** — the escape-print code alone | `bisect-esconly` | `vm-contract` **identity**, `VIRTUAL_APPLICATION_OK`, then `stale` access denied with 25 |

So the **Q-26 wait rewrite alone corrupts every virtual context's pre-HELLO state on silicon**, and
the escape-print code is inert exactly as the short-circuit argument said. Two things follow that
the single-arm boots could not have given:

* the regression is **logic**, not content, size or layout — the content/layout question the
  `esconly` arm existed to answer is closed;
* `esconly` **is** `a56fc415` plus the escape line, so its boot is simultaneously a Tier 1 proof of
  that combination. A board-proven module with the escape line already exists, and the next
  revision does not have to start from `a56fc415` bare.

## Reverted, and the committed source reproduces the proven binary byte-for-byte

The Q-26 rewrite is withdrawn in full: `slot_free_now()` is restored, the wait is
`wait_event_interruptible` on it again, the entry-path `WRITE_ONCE` is gone and both wake sites are
back to a bare `wake_up_all`. Zero references to `slot_free_hint` remain.

**Verified rather than asserted:** rebuilding the committed source yields
`a2a0b2f49d790cc6` — **byte-identical to the module the board proved on Tier 1**. A first attempt at
the revert was semantically equivalent but 48 comment lines and one declaration position different,
and hashed differently; that version was discarded rather than claimed as proven, because a module
of record should reproduce the artifact that was actually booted. The diag tag stays
`bisect-esconly` for the same reason: renaming it would forfeit that byte-identity.

## What the Q-26 rewrite was for, and why it stays reverted

It removed a kernel `WARNING` — `do not call blocking ops when !TASK_RUNNING`, because
`slot_free_now()` takes `vm_lock` inside a `wait_event` condition. That warning is real and the
criticism of the old code stands: a mutex in a wait condition is a latent deadlock hazard.

But the trade is not close. The warning costs one multi-line trace in the ring buffer, once per
boot, on a path that only runs in the sparse tail. The rewrite cost **every virtual context on
silicon**. So the old code stays, the hazard is a known and documented one, and any future fix must
**not touch the entry path** — which is where the rewrite's only per-context statement lived
(`if (!t->started) WRITE_ONCE(slot_free_hint, false);`, immediately after `run()` returns and inside
the interrupt-disabled region).

## Unexplained, and labelled as such

*Why* that rewrite corrupts a pointer read before HELLO is not established. The board lane's
reading — a hint written after the entry returns racing the first resume — is a guess, labelled so
by them, and I have nothing better. The useful constraint is that the only part of the rewrite
executing for a first context is that one post-`run()` store; the wait and the wake-site writes are
on the `CV_THREAD_CREATE` contention path, which a first context does not reach.

**QEMU passed it 34/41 on both pins**, so this is board-only, and it is the third module change in
a row whose silicon behaviour the suite could not predict — see
`RESULT-a-qemu-34-of-41-does-not-license-a-fault-path-default.md`.
