# First stable file-service subset for HostCall v0

This file records the first reusable file-service surface that should sit on top of the
already validated HostCall v0 boundary.

It exists to replace the current proof-only mindset with a small service family that a
future domain-side runtime/libc layer can target.

## Why this is the next step

The current proofs already established:

- shared metadata across two rounds,
- borrowed directional payloads,
- more than one coarse service on the same metadata ABI,
- helper -> domain and domain -> helper payload flow.

So the next unknown is no longer whether one more read-like or write-like demo can work.
The next unknown is whether the project can expose a **stable reusable file boundary** that
moves the design toward SQLite-like workloads without mirroring every libc symbol.

## Design goals

The first stable subset should:

- keep the existing fixed-width metadata block,
- keep the synchronous two-round `call_dom()` flow,
- keep the current metadata ownership rule (`INOUT + SHARED`),
- keep payload access directional and borrowed,
- avoid one-opcode-per-libc-symbol design,
- be reusable by a later domain-side runtime/libc shim,
- be small enough to implement and validate incrementally.

## What should remain local in the domain

These do **not** justify a host service boundary by themselves:

- `memcpy`, `memset`, `memcmp`,
- `strlen`, `strcmp`, `strncmp`,
- simple integer/string formatting helpers,
- local buffering helpers,
- small allocator/runtime glue that does not need an OS object.

Those should remain inside the domain-side runtime/libc layer.

## What should cross to the helper

The first reusable host boundary should focus on operations that really need guest Linux / host-side participation:

- opening or creating a file-like object,
- reading bytes from it,
- writing bytes to it,
- closing it,
- querying basic status needed by higher layers,
- syncing durable state when correctness requires it.

## Recommended first stable subset

### 1. `HC_V0_OP_FILE_OPEN`

Purpose:
- request helper-side open/create semantics for a path-like object.

Request shape:
- payload carries the path bytes and open flags/mode metadata.

Response shape:
- metadata returns a helper-managed file handle token,
- not a raw Linux file descriptor exposed directly to the domain.

Why:
- a reusable runtime needs a stable object identity across later read/write/close calls,
- raw Linux fd values should remain helper-private implementation detail.

### 2. `HC_V0_OP_FILE_READ`

Purpose:
- read bytes from an already opened helper-managed file object.

Request shape:
- metadata identifies the helper-managed handle,
- metadata carries offset/length semantics,
- helper produces response bytes into the payload.

Payload direction:
- helper -> domain,
- `IN + BORROWED` style.

### 3. `HC_V0_OP_FILE_WRITE`

Purpose:
- write bytes from the domain into an already opened helper-managed file object.

Request shape:
- metadata identifies the helper-managed handle,
- domain provides payload bytes and requested offset/length semantics.

Payload direction:
- domain -> helper,
- `OUT + BORROWED` style.

### 4. `HC_V0_OP_FILE_CLOSE`

Purpose:
- release the helper-managed file object and its token.

Why include it early:
- it closes the object-lifetime loop,
- it keeps the first handle-table design honest,
- SQLite-like paths do not want an ever-growing helper-side resource leak.

### 5. `HC_V0_OP_FILE_STAT_BASIC`

Purpose:
- return only the basic file facts that higher layers quickly need.

Initial scope:
- file size,
- existence / type success signal,
- maybe basic mode bits if needed later.

Why this instead of a full `stat(2)` mirror:
- most early consumers do not need the full host struct layout,
- mirroring host libc structs too early makes the ABI fragile.

### 6. `HC_V0_OP_FILE_SYNC`

Purpose:
- force correctness-oriented flush semantics for workloads that care about durability.

Why it matters early:
- SQLite correctness eventually depends on sync-like behavior,
- but the ABI can still keep this as one coarse operation rather than exposing every helper-side subcall.

## What should not be in the first subset

Do **not** start with:

- a full mirror of Linux `open/read/write/lseek/fstat/fsync/flock/mmap/...`,
- one HostCall per libc spelling,
- process management,
- networking,
- directory traversal,
- arbitrary ioctl-style escape hatches,
- direct exposure of host `errno` structs or raw host ABI structs.

Those can come later if the first file subset proves too small.

## Object model recommendation

The helper should own a small handle table.

Recommended rule:

- domain sees only protocol-level file handles/tokens,
- helper maps each token to its own Linux fd and local bookkeeping,
- helper remains free to implement one coarse service with multiple Linux calls,
- the wire ABI stays stable even if the helper implementation changes.

This is the key architectural step that moves the project beyond hardcoded tmpfile demos.

## Recommended metadata evolution

Keep the current metadata block shape if possible.

Minimum additions should be conservative and justified, for example:

- one field for a protocol-level object handle/token,
- reuse existing `offset`, `length`, `result`, `error` fields where possible,
- avoid embedding host-native structs into the shared ABI.

If an extension is needed, prefer a versioned `hostcall_v1` metadata layout rather than an ad hoc pile of per-opcode exceptions.

## SQLite-facing rationale

SQLite-like workloads care about a small number of real OS-facing file behaviors:

- open/create database and journal/WAL files,
- read pages,
- write pages,
- discover file size,
- sync data safely,
- manage object lifetime,
- often coordinate locking later.

That means the first practical bridge toward SQLite is **not** stdout and **not** one more toy tmpfile proof.
It is a reusable file-object path.

## Smallest implementation slice

The smallest code step that matches this design is:

1. add helper-managed file handles,
2. implement `FILE_OPEN`,
3. implement `FILE_READ`,
4. implement `FILE_WRITE`,
5. implement `FILE_CLOSE`,
6. validate one small end-to-end file-object scenario,
7. only then decide whether `FILE_STAT_BASIC` and `FILE_SYNC` are immediately required.

Important current limitation:

- the architectural subset above still makes sense,
- but the current repository does **not** yet treat multi-`PENDING` re-entry from one
  domain invocation as validated baseline behavior,
- so the first implementation may need either one request per domain invocation or a
  separately characterized multi-round control shape until that runtime question is resolved.

## Suggested validation order

### Phase A: handle lifecycle proof
- open a fixed test file,
- receive a helper-managed token,
- close it,
- verify token lifecycle and error paths.

### Phase B: reusable read/write proof
- open,
- write through the handle,
- close,
- reopen,
- read back,
- verify the bytes.

### Phase C: SQLite-facing gap check
- list what is still missing for a tiny SQLite-style file backend shim,
- only then choose whether sync/stat/locking must be added next.

## Exit criterion for this milestone

This milestone is complete when:

- the project has one documented stable file-service subset,
- the helper manages file objects through protocol-level handles,
- the same service family supports both borrowed payload directions,
- the next runtime/libc-facing step can target a reusable service boundary rather than inventing one more demo opcode.

