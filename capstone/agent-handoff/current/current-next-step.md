# Current recommended next step

This file intentionally contains the **current** recommendation only.

## Current recommendation

The reverse-direction payload proof is already validated.
So the next step is **not** another read-like or write-like toy proof.

The next smallest meaningful step is:

> keep the same HostCall v0 control flow and metadata ABI, but stop adding one-off proof opcodes and define the first small stable reusable file-service subset, then implement the first handle-based file-object path on top of it

## Why this is now the right next step

The current runtime baseline already proves:

- the restored OpenSBI/runtime path works again,
- shared-region mutations are visible,
- the same metadata ABI supports more than one coarse HostCall service,
- borrowed payload ownership works in both directions,
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

The first implementation target should be a helper-managed file-object path.

That means:

1. helper owns a file-handle table,
2. domain sees only protocol-level handles/tokens,
3. helper keeps raw Linux file descriptors private,
4. later read/write/close requests target the protocol-level handle.

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

After the subset is documented, the next code change should be the smallest vertical slice that proves the object model:

1. open a helper-side file through the protocol,
2. return a protocol-level handle,
3. write or read through that handle,
4. close it,
5. validate handle lifetime and error handling.

That is the first step that genuinely moves the project closer to SQLite-like file workloads.

## Exit criterion

This milestone is complete when:

- the stable file-service subset is documented,
- the handoff no longer claims that reverse-direction payload is still the next missing proof,
- the first handle-based file-object path is implemented and validated,
- the next runtime/libc-facing step can build on a reusable service boundary instead of inventing another demo opcode.

## What not to regress

Do **not** accidentally drop back to the wrong-firmware path by deleting or bypassing:

- `capstone/caplifive-buildroot/build/local.mk`

If that file disappears again, the older `error=-2` / stock-OpenSBI symptoms are expected to return.
