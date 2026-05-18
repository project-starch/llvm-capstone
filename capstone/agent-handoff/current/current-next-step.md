# Current recommended next step

This file intentionally contains the **current** recommendation only.

## Current recommendation

The reverse-direction payload proof is already validated.
The tree now also has:

- a first helper-managed `FILE_OPEN` / `FILE_CLOSE` lifecycle proof,
- and a first handle-based `FILE_OPEN -> FILE_WRITE -> FILE_CLOSE` proof,
- and a first handle-based `FILE_OPEN -> FILE_READ -> DONE` proof,
- and a first handle-based `FILE_OPEN -> FILE_WRITE -> FILE_SYNC -> FILE_CLOSE` proof,
- and a first composed `FILE_OPEN -> FILE_WRITE -> FILE_SYNC -> FILE_CLOSE -> FILE_OPEN -> FILE_READ -> FILE_CLOSE` proof,

both using explicit revoke-before-reborrow on the reused borrowed payload region.
So the next step is **not** another basic read-like or write-like toy proof.

The next smallest meaningful step is:

> keep the same HostCall v0 control flow and metadata ABI, preserve the now-confirmed borrowed-region revoke discipline, and choose the next post-`FILE_SYNC` consumer-facing semantic from a concrete higher-layer need rather than by adding another proof in the abstract

## Why this is now the right next step

The current runtime baseline already proves:

- the restored OpenSBI/runtime path works again,
- shared-region mutations are visible,
- the same metadata ABI supports more than one coarse HostCall service,
- borrowed payload ownership works in both directions,
- helper-side request snapshotting works in the current two-round HostCall proofs,
- one helper-managed handle can survive across more than one request round and then be closed,
- bytes can now also be written through that helper-managed handle on a later round,
- bytes can now also be read back through that helper-managed handle model,
- that helper-managed file object can now also expose an explicit sync boundary after writes,
- those modular operations now compose into one reusable file-object scenario,
- the domain remains the initiator and the helper remains the executor.

That means the main unresolved question is no longer:

> can one more proof opcode fit the ABI?

It is now:

> which next semantic most usefully increases the correctness envelope of that small reusable service family?

The current answer is no longer "add `FILE_SYNC`".

That step is now done.

The next question is:

> which real higher-layer semantic is still missing after `OPEN`, `READ`, `WRITE`, `SYNC`, and `CLOSE` already work on the same reusable handle-based boundary?

## What this recommendation is trying to avoid

Do **not** drift into either of these extremes:

1. one HostCall opcode per libc symbol,
2. a too-large speculative Linux ABI mirror.

The intended design remains service-oriented:

- keep local computation helpers inside the domain/runtime,
- cross the boundary only for real OS-facing work,
- keep the wire ABI small and stable.

## Concrete next implementation target

The first implementation target is now a partially validated helper-managed
file-object path, and the next remaining gap is clear.

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

The repository now also has one validated concrete example of that rule in a real
composed service path:

- `FILE_OPEN` request,
- helper response with token,
- helper-side `revoke_region()` plus re-share of the same payload region,
- `FILE_WRITE` request using the returned token,
- helper-side `revoke_region()` plus re-share of the same payload region,
- `FILE_SYNC` request using that same token,
- helper-side `revoke_region()` plus re-share of the same payload region,
- or `FILE_READ` request using the returned token,
- helper-side `revoke_region()` plus re-share of the same payload region for the
  response direction,
- `FILE_CLOSE` request,
- final completion.

## Recommended first stable subset

The first stable file-service subset should cover:

- `FILE_OPEN`
- `FILE_READ`
- `FILE_WRITE`
- `FILE_CLOSE`

Then add only if justified by the next consumer:

- `FILE_STAT_BASIC`
- `FILE_SYNC`

The tree now already validates `FILE_SYNC`, so `FILE_STAT_BASIC` remains the leading small
candidate only if the next consumer concretely needs file size / existence / basic type facts.

The full proposal lives in:

- `current/stable-file-service-subset.md`
- `current/hostcall-file-service-v0-wire-spec.md`

## Smallest code slice after the design note

After the subset is documented and the sync-capable composed file-object scenario exists, the
next code change should target the next concrete higher-layer gap:

1. decide whether the first real consumer now needs `FILE_STAT_BASIC`, file size / truncate,
   or a locking-related primitive,
2. prefer the smallest semantic that exposes a new higher-layer capability rather than a new
   proof-only checkpoint,
3. keep the response path on the borrowed payload reuse discipline already validated by the
   read proof,
4. keep using `revoke_region(payload_region_id)` before reusing the borrowed payload region,
5. avoid inventing more proof-only opcodes unless a concrete consumer really needs them.


That should come before claiming a reusable SQLite-facing file service baseline.

## Exit criterion

This milestone is complete when:

- the stable file-service subset is documented,
- the handoff no longer claims that reverse-direction payload is still the next missing proof,
- the current repeated borrowed-payload reuse rule is documented and followed,
- a reusable handle-based path supports `OPEN`, `WRITE`, `READ`, and `CLOSE`,
- `FILE_SYNC` is implemented and validated as the first durability-oriented semantic,
- only then should the first SQLite-facing file-service slice be treated as validated baseline.

## What not to regress

Do **not** accidentally drop back to the wrong-firmware path by deleting or bypassing:

- `capstone/caplifive-buildroot/build/local.mk`

If that file disappears again, the older `error=-2` / stock-OpenSBI symptoms are expected to return.
