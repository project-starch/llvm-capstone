# The reading that was claimed and never taken: 0 of 4 caught, with the revoker proved to fire

Stock CheriBSD purecap, **libc revocation ON**, `guest_default: preserved`. **7 of 7 cases passed,
runner exit 0.**

This bundle exists because of a retraction landed earlier the same day: `18a16640682b` asserted that
these four cases had been run on stock CheriBSD with revocation enabled and had completed, and no
such run existed. **The predicted outcome was right. What was wrong was asserting it had been
observed.** It has now been observed.

| case | corpus case | reading |
|---|---|---|
| `pool-0-36` | 0 `461fb22053` | `FF2 status=0 events=1 metadata=1472 payload=128` — completes |
| `pool-0-37` | 1 `1886c3269d` | `FF2 status=0 events=1 metadata=4800 payload=512` — completes |
| `pool-0-38` | 2 `316531e61c` | `FF2 status=0 events=1 metadata=1344 payload=128` — completes |
| `pool-0-39` | 3 `5c66a3ab51` | `FF2 status=0 events=1 metadata=384 payload=320` — completes |

**0 of 4 caught.** A buffer returned to an `AVBufferPool` never reaches `free()`, so it never enters
the quarantine CheriBSD's revoker sweeps and the sweep has nothing to find.

## The control fired, and that is the whole difference from the retracted claim

```
REVOCATION_CONTROL revocation=1 tag_after_sweep=0 reissued=1
SUPERVISE expect mc_defect_read 0x101dc0
SUPERVISE fault signal=34 code=2 addr=0x101dc0 pc=0x101dc0
```

The control mallocs a block, frees it, asks the revoker to sweep, re-allocates the same size and
reads through the **old** pointer. It reports `tag_after_sweep=0` — the stale capability lost its tag
— and then **faults**: `SIGPROT`, `si_code` **2** (`PROT_CHERI_TAG`), at an address equal to the one
`supervise` resolved for the labelled probe independently of the run.

So the revoker demonstrably sweeps in this guest. A completion in the four cases therefore means
**"the mechanism is active and did not fire"**, not "the mechanism is absent" — which is precisely
the distinction the withdrawn claim asserted without evidence, and the bar the retraction set.

| platform control | reading |
|---|---|
| `cheribsd-abi` | `CHERI_ABI pointer_bytes=16 runtime_revocation=1` — **PASS** |
| `cheribsd-bounds` | `CHERI_BOUNDARY_READY` — **PASS** |

## What made this runnable at all

Nothing needed inventing. The port's purecap build — `cmake --preset cheribsd` on
`ports/ffmpeg/buffer-pool` — produces `bin/pool-security`, which already registers these four as
cases 36-39. They were run through `ports/common/host/cheribsd/run.py` with
`--runtime-revocation on`, rather than through the port's own PoisonCap wrapper, which passes the
unconditional literal `"off"` at `poisoncap/run.py:129-130` and is the reason the retracted claim's
cited case names could never have carried a revocation-on reading. The control is the memcached
corpus's `revocation-control.c`, reused unchanged because it exports its probe as a global label.

## What this does and does not establish

- **Does:** 0 of 4 caught under revocation on, with the revoker proved to fire in the same boot and
  both platform controls passing.
- **Does:** close the gap the retraction opened. FFmpeg now contributes **4 measured temporal rows**
  where it contributed none this morning.
- **Does not** measure PoisonCap, which is unavailable on this host. The PoisonCap arms citing
  `pool-{0,2}-{36,37,38}` still have no committed bundle — a separate, weaker gap flagged in the
  corpus README and not closed here.
- **N = 1 per cell.**

Files: `result-lines.txt`, `inputs.json`.
