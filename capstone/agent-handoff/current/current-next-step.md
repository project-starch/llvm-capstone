# Current recommended next step

This file intentionally contains the **current** recommendation only.

## Current recommendation

The reverse-direction payload proof is already validated.
The current helper-side proofs now also snapshot shared metadata requests before
service, so the next step is **not** another basic read-like or write-like toy proof.

The next smallest meaningful step is:

> keep the same HostCall v0 control flow and metadata ABI, but apply the now-confirmed borrowed-region revoke discipline before any repeated borrowed payload re-share, then continue the smallest handle-based file-service experiment on top of that rule

## Why this is now the right next step

The current runtime baseline already proves:

- the restored OpenSBI/runtime path works again,
- shared-region mutations are visible,
- the same metadata ABI supports more than one coarse HostCall service,
- borrowed payload ownership works in both directions,
- helper-side request snapshotting works in the current two-round HostCall proofs,
- the domain remains the initiator and the helper remains the executor.

That means the main unresolved question is no longer:

> can one more proof opcode fit the ABI?

It is now:

> can this boundary support a small reusable service family that a future runtime/libc layer can actually target?

## What this recommendation is trying to avoid

Do **not** drift into either of these extremes:

1. one HostCall opcode per libc symbol,
2. a too-large speculative Linux ABI mirror.

The intended design remains service-oriented:

- keep local computation helpers inside the domain/runtime,
- cross the boundary only for real OS-facing work,
- keep the wire ABI small and stable.

## Concrete next implementation target

The first implementation target is still expected to become a helper-managed
file-object path, but the diagnostics now show a more specific runtime fact.

The repository now has source-backed evidence that:

1. a second successive `PENDING` can work when the probe is metadata only,
2. but reusing/re-sharing the same borrowed output payload for the next round
   reproduces the QEMU assertion in `helper_csmrev`.

An authoritative runtime/QEMU reply now also confirms the intended invariant:

> once a region is borrow-shared, it must be revoked before anything else can be
> done to it, including borrow-sharing it again.

That means:

1. the stable file-service subset remains the right architectural target,
2. generic multi-`PENDING` support is no longer the blocker,
3. future multi-op file-object work should explicitly `revoke_region()` before any
   repeated borrowed output payload re-share,
4. a runtime-side auto-normalization patch is optional ergonomics work, not the
   required correctness fix for the current branch.

## Recommended first stable subset

The first stable file-service subset should cover:

- `FILE_OPEN`
- `FILE_READ`
- `FILE_WRITE`
- `FILE_CLOSE`

Then add only if justified by the next consumer:

- `FILE_STAT_BASIC`
- `FILE_SYNC`

The full proposal lives in:

- `current/stable-file-service-subset.md`

## Smallest code slice after the design note

After the subset is documented, the next code change should be the smallest
real helper/service path that obeys this now-known rule:

1. if the same payload region will be borrowed again in a later round, call
   `revoke_region(payload_region_id)` first,
2. then re-share it as borrowed for the next round,
3. keep the non-revoke probe as the negative diagnostic proving why that discipline exists.

That should come before promoting a multi-op handle path to the validated baseline.

## Exit criterion

This milestone is complete when:

- the stable file-service subset is documented,
- the handoff no longer claims that reverse-direction payload is still the next missing proof,
- the current repeated borrowed-payload reuse rule is documented and followed,
- only then should the first handle-based file-object path be treated as validated baseline.

## What not to regress

Do **not** accidentally drop back to the wrong-firmware path by deleting or bypassing:

- `capstone/caplifive-buildroot/build/local.mk`

If that file disappears again, the older `error=-2` / stock-OpenSBI symptoms are expected to return.
