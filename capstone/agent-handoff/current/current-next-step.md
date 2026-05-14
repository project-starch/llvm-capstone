# Current recommended next step

This file intentionally contains the **current** recommendation only.

## Current recommendation

The reverse-direction payload proof is already validated.
The current helper-side proofs now also snapshot shared metadata requests before
service, so the next step is **not** another basic read-like or write-like toy proof.

The next smallest meaningful step is:

> keep the same HostCall v0 control flow and metadata ABI, but characterize the newly observed multi-`PENDING` limitation before treating a handle-based multi-op file-object path as safe validated baseline behavior

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
file-object path, but one newly observed runtime fact now matters first.

An attempted multi-op proof that tried to do:

1. `FILE_OPEN` request,
2. helper response,
3. second domain `PENDING` for `FILE_WRITE`,

hit a QEMU assertion in the current environment during the attempted re-entry after
the first response. So the repository does **not** yet have source-backed evidence
that one domain invocation may safely emit multiple successive `PENDING` returns.

That means:

1. the stable file-service subset remains the right architectural target,
2. but the next implementation step should first determine whether multi-`PENDING`
   is a supported runtime shape,
3. or whether the service family must initially stay within the already validated
   one-`PENDING`-then-complete pattern.

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
diagnostic that answers this precise runtime question:

1. can one domain invocation safely return `PENDING` more than once,
2. or does the current runtime/QEMU path only support the already validated two-round shape?

That answer should come before promoting a multi-op handle path to the validated baseline.

## Exit criterion

This milestone is complete when:

- the stable file-service subset is documented,
- the handoff no longer claims that reverse-direction payload is still the next missing proof,
- the current multi-`PENDING` limitation is either explained or lifted,
- only then should the first handle-based file-object path be treated as validated baseline.

## What not to regress

Do **not** accidentally drop back to the wrong-firmware path by deleting or bypassing:

- `capstone/caplifive-buildroot/build/local.mk`

If that file disappears again, the older `error=-2` / stock-OpenSBI symptoms are expected to return.
