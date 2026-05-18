# Practical HostCall file-service v0 wire spec

This file turns the higher-level file-service subset note into an implementation-oriented
wire spec that the next helper/domain code can follow directly.

It is intentionally conservative:

- keep the existing `hostcall_v0` metadata block,
- keep one metadata region plus one payload region,
- keep the already validated synchronous `PENDING -> RESP -> DONE` round shape,
- keep helper-managed file handles,
- and apply the now-confirmed revoke-before-reborrow rule whenever the same borrowed
  payload region is reused on a later round.

## Executive summary

The modular file-service ABI should reserve `FILE_OPEN`, `FILE_READ`, `FILE_WRITE`,
and `FILE_CLOSE` from the start.

Rationale:

- SQLite-like consumers need both read and write directions,
- the read path is where the payload-direction flip is most visible,
- freezing it now avoids redesigning the payload contract immediately after landing
  the first `OPEN` / `WRITE` / `CLOSE` code path.

However, the **first code landing** does not need to implement every opcode at once.
A practical staged order is:

1. `FILE_OPEN`
2. `FILE_CLOSE`
3. `FILE_WRITE`
4. `FILE_READ`
5. `FILE_SYNC`
6. only then consider `FILE_STAT_BASIC`

That keeps the initial code slice small while still freezing a reusable modular ABI.

## 1. Non-goals

This spec is **not** trying to provide:

- a raw Linux syscall mirror,
- host-visible register-level yield/resume ABI,
- direct exposure of helper-side file descriptors,
- a full hosted libc ABI,
- or a one-off `open-write-close` super-protocol.

The intended model is modular:

- `FILE_OPEN`, `FILE_READ`, `FILE_WRITE`, `FILE_CLOSE` are separate operations,
- a later runtime/libc/sqlite shim may compose them as needed,
- e.g. `OPEN -> WRITE -> WRITE -> CLOSE`, `OPEN -> READ -> CLOSE`,
  `OPEN -> READ -> WRITE -> CLOSE`, etc.

## 2. Why the protocol keeps metadata and payload separate

The split is intentional, not accidental.

### 2a. Different lifetime and mutation rules

The metadata block contains:

- protocol state (`phase`),
- operation selection (`opcode`),
- generic payload addressing (`offset`, `length`),
- generic response status (`result`, `error`).

Those fields are small, fixed-width, and are touched in almost every round.

The payload region contains:

- variable-size request-specific arguments,
- path bytes,
- write data,
- or read response bytes.

Keeping them separate lets the protocol keep one small long-lived control structure while
re-sharing the larger data region with direction-specific borrowed permissions.

### 2b. Different permission models

The metadata region intentionally stays `INOUT + SHARED` across the whole interaction.
That is convenient because both sides need to update it.

The payload region is where least-privilege matters:

- `FILE_OPEN` / `FILE_WRITE` request bytes are domain -> helper,
- `FILE_READ` response bytes are helper -> domain.

If metadata and payload were collapsed into one region, the simplest safe sharing mode would
often degrade back toward broad `INOUT + SHARED` access for everything. That would weaken the
directional borrowed model that the current runtime proofs are trying to validate.

### 2c. This is not meant to model one atomic kernel syscall packet

The protocol is a synchronous shared-memory RPC, not a single indivisible kernel entry.
Its atomicity boundary is the round transition:

1. domain writes request state,
2. domain returns `PENDING`,
3. helper snapshots what it needs immediately,
4. helper performs the work,
5. helper writes response state,
6. helper re-enters the domain.

That means the design should be judged like a disciplined RPC protocol, not like a literal
one-shot `syscall` trap frame.

### 2d. TOCTOU risk exists, but the split itself is not the root cause

TOCTOU appears if one side keeps re-reading mutable shared state after the handoff point.
The current mitigation is explicit protocol discipline:

- snapshot metadata immediately after `call_dom()` returns,
- snapshot request-side payload bytes immediately if the helper will rely on them,
- do not trust later re-reads of mutable request state,
- for read responses, snapshot request header first, then re-share payload for the response
  direction if needed.

So the answer is not "merge metadata and payload into one blob".
The answer is "keep the shared-memory protocol narrow and snapshot the fields that become
authoritative for the current round".

## 3. Shared metadata remains unchanged

Keep the existing fixed-width shared metadata layout:

```c
struct hostcall_v0 {
    uint64_t phase;
    uint64_t opcode;
    uint64_t offset;
    uint64_t length;
    int64_t  result;
    int64_t  error;
};
```

### Field meaning in this file-service spec

- `phase`: protocol state (`REQ`, `RESP`, `DONE`, `ERROR`)
- `opcode`: which file operation is requested
- `offset`: byte offset **inside the payload region** where variable-length data starts
- `length`: byte length of that variable-length data
- `result`: operation result in the response phase
- `error`: `0` or negative errno-like failure code in the response phase

### Important request-phase rule

During `REQ`:

- `result` must be `0`
- `error` must be `0`

Any per-operation request arguments beyond `offset`/`length` live in the payload header,
not in overloaded metadata fields.

That keeps `result` and `error` unambiguously response-only.

## 4. Recommended opcode values

The current probe header already uses low opcode values for existing proofs.
To avoid churn with those diagnostics, reserve a separate block for the file-service ABI:

```c
#define HC_V0_OP_FILE_OPEN        16ULL
#define HC_V0_OP_FILE_READ        17ULL
#define HC_V0_OP_FILE_WRITE       18ULL
#define HC_V0_OP_FILE_CLOSE       19ULL
#define HC_V0_OP_FILE_STAT_BASIC  20ULL   /* reserved for later */
#define HC_V0_OP_FILE_SYNC        21ULL
```

These values are a recommendation for the next implementation patch series.

## 5. Region contract

## 5a. Metadata region

Always share the metadata region as:

```c
shared_region_annotated(dom_id, metadata_region_id,
                        CAPSTONE_ANNOTATION_PERM_INOUT,
                        CAPSTONE_ANNOTATION_REV_SHARED);
```

Rationale:

- this already matches the validated HostCall baseline,
- both sides need to mutate `phase`, `result`, and `error`,
- metadata is intentionally long-lived across the interaction.

## 5b. Payload region

Use one payload region, but re-share it with the direction needed for the next round.

### Request payload from domain to helper

For `FILE_OPEN` path bytes and `FILE_WRITE` data bytes, share as:

```c
shared_region_annotated(dom_id, payload_region_id,
                        CAPSTONE_ANNOTATION_PERM_OUT,
                        CAPSTONE_ANNOTATION_REV_BORROWED);
```

### Response payload from helper to domain

For `FILE_READ` response bytes, share as:

```c
shared_region_annotated(dom_id, payload_region_id,
                        CAPSTONE_ANNOTATION_PERM_IN,
                        CAPSTONE_ANNOTATION_REV_BORROWED);
```

## 5c. Revoke-before-reborrow rule

If the same payload region ID will be borrowed again on a later round, the helper must do:

```c
revoke_region(payload_region_id);
shared_region_annotated(dom_id, payload_region_id, ... , CAPSTONE_ANNOTATION_REV_BORROWED);
```

This is required even when the direction stays the same (`OUT -> OUT`), and also when the
helper flips the direction (`OUT -> IN`).

That rule is now source-backed by:

- the negative payload-reuse probe,
- the positive explicit-revoke probe,
- and the runtime/QEMU clarification that an already borrow-shared region must be revoked
  before doing anything else with it.

## 6. Helper-managed handle table

The helper owns a small mapping:

```c
token -> { linux_fd, flags, optional bookkeeping }
```

Rules:

- the domain only sees `token`
- the token is never a raw Linux fd
- the helper may recycle tokens after `FILE_CLOSE`
- token `0` should be treated as invalid / reserved

Recommended first implementation:

- use a small fixed-size array of slots,
- linearly scan for a free slot,
- return `-EMFILE` or `-ENFILE` style failure when full.

So in the initial implementation this is **not** a hash table.
It is just a compact slot table keyed by a small integer token.

If a later workload needs thousands of live handles, the helper implementation may switch to a
different internal data structure without changing the wire ABI, because the domain still sees
only the token.

## 7. Fixed-size request headers stored in the payload region

This section describes the fixed-size request-specific headers that live at the beginning of the
payload region.

The metadata `offset` points to the start of the variable-length byte area.

That is why these layouts live in payload rather than metadata:

- they are operation-specific rather than generic,
- some operations need variable-length trailing bytes,
- and keeping them in payload avoids turning `hostcall_v0` into an opcode-specific union.

### 7a. `FILE_OPEN`

```c
struct hc_file_open_req_v0 {
    uint64_t flags;
    uint64_t mode;
    char path[];
};
```

Request contract:

- the fixed-size header is exactly the bytes before `path[]`
- in C terms, the header size is `offsetof(struct hc_file_open_req_v0, path)`
- because `path[]` is a flexible array member, not a pointer field, it contributes **zero**
  bytes to the fixed-size header
- payload bytes `[0, offsetof(struct hc_file_open_req_v0, path))` contain `flags` and `mode`
- path bytes start at `metadata.offset = offsetof(struct hc_file_open_req_v0, path)`
- `metadata.length = path_len`
- path bytes do not need a trailing `\0` in the wire format; helper may copy and append one locally

Practical note:

- `sizeof(struct hc_file_open_req_v0)` is commonly equal to `offsetof(..., path)` for a
  flexible-array-member struct on mainstream compilers,
- but the protocol text should prefer the `offsetof(..., path)` wording because it states the
  intent directly and avoids confusion with pointer-sized trailing fields.

Response contract:

- `metadata.result = handle_token` on success
- `metadata.error = 0` on success
- `metadata.result = -1` and `metadata.error = -errno` on failure

### 7b. `FILE_WRITE`

```c
struct hc_file_write_req_v0 {
    uint64_t handle;
    uint64_t file_offset;
    uint64_t flags;      /* reserved, write as 0 for now */
    uint64_t reserved0;
    uint8_t  data[];
};
```

Request contract:

- header starts at payload offset `0`
- write data starts at `metadata.offset = offsetof(struct hc_file_write_req_v0, data)`
- `metadata.length = data_len`
- `file_offset` is the logical file offset to write at
- `flags` is reserved for later extensions such as append/partial-write policy; write `0` for now

Response contract:

- `metadata.result = bytes_written`
- `metadata.error = 0` on success
- `metadata.result = -1` and `metadata.error = -errno` on failure

### 7c. `FILE_READ`

```c
struct hc_file_read_req_v0 {
    uint64_t handle;
    uint64_t file_offset;
    uint64_t flags;      /* reserved, write as 0 for now */
    uint64_t reserved0;
    uint8_t  data[];
};
```

Request contract:

- header starts at payload offset `0`
- `metadata.offset = offsetof(struct hc_file_read_req_v0, data)`
- `metadata.length = max_bytes_requested`
- the request header is written by the domain before the first `PENDING`
- the helper snapshots the header immediately after return, then may overwrite the payload
  byte area with response data for the response round

Response contract:

- helper writes up to `metadata.length` bytes into `payload[metadata.offset ...]`
- helper sets `metadata.result = bytes_read`
- helper sets `metadata.error = 0` on success
- on failure, helper sets `metadata.result = -1` and `metadata.error = -errno`

### 7d. `FILE_CLOSE`

```c
struct hc_file_close_req_v0 {
    uint64_t handle;
};
```

Request contract:

- header starts at payload offset `0`
- `metadata.offset = 0`
- `metadata.length = 0`

Response contract:

- `metadata.result = 0` on success
- `metadata.error = 0` on success
- `metadata.result = -1` and `metadata.error = -errno` on failure

### 7e. `FILE_SYNC`

```c
struct hc_file_sync_req_v0 {
    uint64_t handle;
    uint64_t flags;
};
```

Request contract:

- header starts at payload offset `0`
- `metadata.offset = 0`
- `metadata.length = 0`
- `flags = 0` requests the first conservative durability-oriented behavior: `fsync(fd)`

Response contract:

- `metadata.result = 0` on success
- `metadata.error = 0` on success
- `metadata.result = -1` and `metadata.error = -errno` on failure

## 8. What lives in metadata vs payload

### Metadata

Metadata carries only the generic control-plane fields that every operation needs:

- `phase`
- `opcode`
- `offset`
- `length`
- `result`
- `error`

This block is intentionally small, fixed-width, and stable across all operations.

### Payload

Payload carries the operation-specific data-plane bytes:

- `FILE_OPEN`: fixed-size open header plus path bytes
- `FILE_WRITE`: fixed-size write header plus write data bytes
- `FILE_READ`: fixed-size read header, then later response data bytes
- `FILE_CLOSE`: fixed-size close header only

That separation keeps `hostcall_v0` generic while still allowing operation-specific request
formats and variable-length data.

## 9. Operation state machine

Each operation remains one ordinary HostCall request/response pair.
The project does **not** need a special monolithic protocol for `OPEN/WRITE/CLOSE` as a
single giant transaction.

Instead, a later domain runtime layer should expose helpers such as:

```c
long hc_file_open(...);
long hc_file_write(...);
long hc_file_read(...);
long hc_file_close(...);
```

and compose them however the caller needs.

### Generic round shape

1. domain prepares metadata + payload request
2. domain executes `DOM_RETURN(HC_V0_RET_PENDING)`
3. helper snapshots request metadata immediately
4. helper snapshots payload header / bytes immediately if needed
5. helper performs ordinary Linux work
6. helper writes response metadata (and response payload for read)
7. helper calls the domain again
8. domain consumes the response
9. domain returns `HC_V0_RET_DONE` for this operation

That means modularity comes from **repeated use of the same small operation ABI**, not from
writing a separate wrapper protocol for every possible open/read/write/close combination.

## 10. Direction-specific notes by opcode

### `FILE_OPEN`

- request payload direction: domain -> helper
- response payload: none
- metadata only is sufficient for the response

### `FILE_WRITE`

- request payload direction: domain -> helper
- response payload: none
- helper responds with written-byte count in metadata

### `FILE_READ`

- request header direction: domain -> helper
- response data direction: helper -> domain
- if the same payload region is reused for the response, helper must snapshot the request,
  then `revoke_region(payload_region_id)`, then re-share the region as `IN + BORROWED`
  before the response round

Practical note for a one-payload composed scenario:

- if the protocol also needs the domain to stage another request immediately after
  consuming the read response on that same payload region, the helper may need a
  slightly broader borrowed re-share for that one handoff (for example `INOUT + BORROWED`)
  unless the ABI grows a second payload region or another way to stage the follow-on request.

### `FILE_CLOSE`

- request payload direction: domain -> helper
- response payload: none
- metadata only is sufficient for the response

### `FILE_SYNC`

- request payload direction: domain -> helper
- response payload: none
- metadata only is sufficient for the response
- the initial conservative helper behavior is `flags == 0 -> fsync(fd)`

## 11. Error model

Recommended response rule:

- `metadata.error == 0` means success
- `metadata.error < 0` means failure and carries a negative errno-like code
- `metadata.result` carries the success return value when `error == 0`
- `metadata.result` should be `-1` when `error != 0`

Examples:

- `FILE_OPEN` success -> `result = token`, `error = 0`
- `FILE_OPEN` failure -> `result = -1`, `error = -ENOENT`
- `FILE_WRITE` success -> `result = bytes_written`, `error = 0`
- `FILE_CLOSE` success -> `result = 0`, `error = 0`

## 12. Why the initial wire spec already reserves `FILE_READ`

Reasons:

1. SQLite-like consumers need both read and write semantics.
2. `FILE_READ` is where the payload-direction reversal is most explicit, so it is the best
   place to freeze the revoke-before-reborrow discipline precisely.
3. If the spec shipped only `OPEN/WRITE/CLOSE`, the very next step would need another
   wire-format discussion for read-side response bytes.
4. Including `READ` in the spec does **not** force the first code patch to land it on day one.

So the recommended split is:

- freeze `OPEN/READ/WRITE/CLOSE` in the design now,
- implement them in small staged patches.

## 13. Recommended first implementation sequence

### Phase 1: shared definitions and helper table

- add file-service opcode constants
- add payload request structs in one shared header
- add helper-side token table utilities

### Phase 2: `FILE_OPEN` / `FILE_CLOSE` proof

Validate:

- open a fixed guest path through helper Linux userspace
- return token
- close token
- reject invalid token on second close

Why first:

- establishes object lifetime and error handling without response payload complexity

Current status:

- this phase now exists in-tree as a validated `FILE_OPEN` / `FILE_CLOSE` proof,
- it exercises helper-managed handle allocation and release,
- and it reuses the same borrowed payload region across the later close request via
  explicit revoke-before-reborrow.

### Phase 3: `FILE_WRITE` proof

Validate:

- open target path
- write request payload bytes through token
- close
- confirm helper-side file contents from guest Linux userspace

Why next:

- request-side borrowed payload is already a validated pattern in the tree

Current status:

- this phase now exists in-tree as a validated handle-based `FILE_WRITE` proof,
- it exercises `FILE_OPEN -> FILE_WRITE -> FILE_CLOSE` on one domain invocation,
- and it uses explicit revoke-before-reborrow between the open, write, and close
  request rounds.

### Phase 4: `FILE_READ` proof

Validate:

- open source path
- request read through token
- helper writes response bytes into payload
- helper performs revoke-before-reborrow before the response round if reusing the same payload region
- domain verifies bytes
- close

Why after write:

- it adds the direction flip and exercises the new rule most directly

Current status:

- this phase now exists in-tree as a validated handle-based `FILE_READ` proof,
- it exercises `FILE_OPEN -> FILE_READ -> DONE` on one domain invocation,
- and it snapshots the read request header before revoking and re-sharing the same
  payload region as borrowed input for the response round.

### Phase 5: decide whether `STAT_BASIC` / `SYNC` are immediately needed

Do not add them by default until a real consumer needs them.

Current status:

- the tree now has the first composed reusable file-object scenario:
- the tree now also has the first focused handle-based `FILE_SYNC` proof:
  `FILE_OPEN -> FILE_WRITE -> FILE_SYNC -> FILE_CLOSE`,
- the tree now has the first composed reusable file-object scenario:
  `FILE_OPEN -> FILE_WRITE -> FILE_SYNC -> FILE_CLOSE -> FILE_OPEN -> FILE_READ -> FILE_CLOSE`,
- that proof reuses one metadata region and one payload region across the whole
  scenario,
- and for the final `FILE_READ -> FILE_CLOSE` handoff it uses a slightly broader
  borrowed response share so the domain can consume the read bytes and then stage
  the last close request without a second payload region.

## 14. Minimal validation scenarios

### Scenario A: handle lifecycle

- `FILE_OPEN`
- `FILE_CLOSE`
- second `FILE_CLOSE` on same token should fail

### Scenario B: write path

- `FILE_OPEN`
- `FILE_WRITE`
- `FILE_CLOSE`
- verify bytes from helper-side Linux file view

### Scenario C: read path

- `FILE_OPEN`
- `FILE_READ`
- `FILE_CLOSE`
- verify bytes inside the domain

### Scenario D: mixed modular composition

- `FILE_OPEN`
- `FILE_WRITE`
- `FILE_WRITE`
- `FILE_READ`
- `FILE_CLOSE`

This is the first scenario that demonstrates the intended modular composition model.

## 15. What should remain for a later `hostcall_v1`

Do **not** force a metadata redesign immediately.

Move to `hostcall_v1` only if experience shows that `v0` becomes too awkward.
Examples that would justify a versioned extension later:

- a dedicated metadata handle field becomes clearly worth it
- multi-buffer operations are needed
- richer stat/sync/locking semantics need more structured fixed-width arguments
- response metadata can no longer stay generic without excessive opcode-specific exceptions

Until then, `hostcall_v0 + payload headers + helper-managed handle table` is the preferred
practical implementation path.

