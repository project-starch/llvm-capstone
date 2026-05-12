# Split host-enclave strategy: source-backed summary

This note records the current source-backed architectural recommendation for the Capstone project.

## Short version

For the near-term milestone, prefer:

> **split host-enclave execution with synchronous shared-memory RPC**

and not:

> full native hosted `capstone64-unknown-linux-gnu` userspace bring-up

and not yet:

> resumable yield/resume syscall proxy with userspace-visible register ABI

## What is already supported by the repository

### 1. Userspace can already create/call domains and create/share regions

Relevant file:
- `capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c`

Relevant functions:
- `call_dom()`
- `create_region()`
- `map_region()`
- `shared_region_annotated()`
- `revoke_region()`

This is enough substrate for a shared metadata region plus a shared payload/bounce-buffer region.

### 2. Multi-round host/domain interaction already exists

Relevant file:
- `capstone/caplifive-buildroot/package/modcapstone/userspace/miniweb_frontend.c`

Observed pattern:
- host shares regions with the domain
- host calls the domain
- domain returns
- host performs ordinary Linux work (`open/read/write/close`)
- host calls the same domain again

So the system already supports a state-machine style RPC protocol over repeated `call_dom()` rounds.

### 3. Domain return path already exists in SBI/runtime

Relevant file:
- `capstone/caplifive-buildroot/components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.c`

Important function:
- `return_from_domain()` writes a return value and transfers control back via `__domreturnsaves(...)`

### 4. A general userspace-visible resume ABI is not currently evident

Relevant file:
- `capstone/caplifive-buildroot/package/modcapstone/include/capstone.h`

Currently visible ioctls include:
- `IOCTL_DOM_CREATE`
- `IOCTL_DOM_CALL`
- region operations

What is **not** visible there:
- no `DOM_RESUME`
- no explicit register snapshot ABI
- no general userspace trapframe export ABI

Therefore, the safest architecture today is a synchronous state-machine protocol over shared regions, not a speculative SGX-style yield/resume design.

## Recommended HostCall ABI v0

Two shared regions:

1. **metadata region**
2. **payload region**

Suggested metadata layout:

```c
struct hostcall_v0 {
    uint64_t phase;   // INIT / REQ / RESP / DONE / ERROR
    uint64_t opcode;  // HC_WRITE_STDOUT = 1
    uint64_t offset;  // into payload region
    uint64_t length;  // request length in bytes
    int64_t  result;  // host return value
    int64_t  error;   // errno-like code
};
```

### Concrete v0 engineering spec

This section freezes the first experiment more precisely, based on what is
already visible in the repository.

#### A. Exact metadata layout

Use a fixed-width 48-byte structure, even though much of the current codebase
often uses `unsigned long`.

```c
struct hostcall_v0 {
    uint64_t phase;   // offset 0x00
    uint64_t opcode;  // offset 0x08
    uint64_t offset;  // offset 0x10
    uint64_t length;  // offset 0x18
    int64_t  result;  // offset 0x20
    int64_t  error;   // offset 0x28
};
```

Recommended constant offsets:

```c
#define HC_V0_PHASE_OFFSET   0x00
#define HC_V0_OPCODE_OFFSET  0x08
#define HC_V0_DATA_OFFSET    0x10
#define HC_V0_LENGTH_OFFSET  0x18
#define HC_V0_RESULT_OFFSET  0x20
#define HC_V0_ERROR_OFFSET   0x28
#define HC_V0_STRUCT_SIZE    0x30
```

Why fixed-width types are preferred here:
- the host userspace ABI currently returns only one `unsigned long` through
  `call_dom()`
- existing Capstone userspace examples assert that `sizeof(long) == 8`
- freezing the shared-memory ABI explicitly as 64-bit fields makes the design
  less fragile and easier to port into a later micro-libc layer

#### B. Exact phase / opcode / return-status values

Use separate namespaces for:
- shared-memory state (`phase`)
- request kind (`opcode`)
- scalar `call_dom()` return code

That separation keeps the protocol easy to debug and avoids overloading the
single scalar return channel.

Recommended values:

```c
/* metadata.phase */
#define HC_V0_PHASE_INIT   0
#define HC_V0_PHASE_REQ    1
#define HC_V0_PHASE_RESP   2
#define HC_V0_PHASE_DONE   3
#define HC_V0_PHASE_ERROR  4

/* metadata.opcode */
#define HC_V0_OP_NONE          0
#define HC_V0_OP_WRITE_STDOUT  1

/* DOM_RETURN / call_dom() scalar status */
#define HC_V0_RET_DONE     0
#define HC_V0_RET_PENDING  1
#define HC_V0_RET_ERROR    2
```

Recommended ownership rules:
- the **domain** writes `phase = REQ` before the first `DOM_RETURN(HC_V0_RET_PENDING)`
- the **host** writes `phase = RESP` after servicing the request
- the **domain** writes `phase = DONE` before its final `DOM_RETURN(HC_V0_RET_DONE)`
- either side may write `phase = ERROR` and return `HC_V0_RET_ERROR` if an invariant is violated

#### C. Exact recommended region annotations

The repository already shows:
- `CAPSTONE_ANNOTATION_PERM_INOUT` / `CAPSTONE_ANNOTATION_REV_SHARED` for shared metadata-like regions
- `CAPSTONE_ANNOTATION_PERM_IN` or `OUT` / `CAPSTONE_ANNOTATION_REV_BORROWED` for directional borrowed buffers

Source-backed observations:
- `miniweb_frontend.c` shares its metadata region as `INOUT + SHARED`
- `shared_region_annotated()` in OpenSBI implements:
  - `REV_SHARED` = post-return revoke disabled
  - `REV_BORROWED` = post-return revoke enabled

Recommended policy for the **very first proof-of-concept**:

1. **metadata region**
   - `annotation_perm = CAPSTONE_ANNOTATION_PERM_INOUT`
   - `annotation_rev  = CAPSTONE_ANNOTATION_REV_SHARED`

2. **payload region**
   - conservative / simplest first experiment:
     - `annotation_perm = CAPSTONE_ANNOTATION_PERM_INOUT`
     - `annotation_rev  = CAPSTONE_ANNOTATION_REV_SHARED`
   - tighter follow-up version after the PoC works:
     - `annotation_perm = CAPSTONE_ANNOTATION_PERM_OUT`
     - `annotation_rev  = CAPSTONE_ANNOTATION_REV_BORROWED`

Why start with `INOUT + SHARED` for both in v0:
- it minimizes surprises from automatic revoke behavior while the protocol is
  still being validated
- it allows both sides to inspect the same memory across both `call_dom()` rounds
- it matches the already proven metadata-sharing pattern in the repository

Why tighten the payload later:
- for `WRITE_STDOUT`, the domain only needs to **write** the payload
- the host only needs to **read** it after return
- therefore `OUT + BORROWED` is a better least-privilege end state once the basic flow is proven

#### D. Region discovery convention inside the domain

The current runtime already supports querying region count and region base from
inside the domain via:
- `SBI_EXT_CAPSTONE_REGION_COUNT`
- `SBI_EXT_CAPSTONE_REGION_QUERY`

The `miniweb` backend infers newly shared regions from the tail of the region ID
space using `region_count - N` ordering.

Recommended convention for v0:
- host creates the domain first
- then host creates exactly two regions in this order:
  1. metadata
  2. payload
- then host shares them with the domain in the same order
- domain startup computes:
  - `region_n = REGION_COUNT()`
  - `metadata_region = region_n - 2`
  - `payload_region  = region_n - 1`

This should be treated as a **v0 convention**, not a forever ABI promise.
It is good enough for the first proof because it matches the pattern already
used in the repository and avoids inventing another bootstrap channel.

#### E. Recommended implementation base in the current tree

For the first experiment, the most natural starting point is:

- host side:
  - derive from `capstone/caplifive-buildroot/package/modcapstone/userspace/sbi-dom.c`
- domain side:
  - reuse the existing `sbi.dom` + `sbi.smode` style split already used by
    `create_dom("/test-domains/sbi.dom", "/test-domains/sbi.smode")`

Why this is the safest base:
- `sbi-dom.c` already demonstrates a two-round `call_dom()` host flow
- `sbi.smode.c` already demonstrates repeated `DOM_RETURN` from S-mode
- `sbi.dom.c` already provides the C-domain side reentry substrate used to host
  the S-mode component

## Recommended first proof-of-concept

### Host side

- create domain
- create metadata region
- create payload region
- map both regions
- share both into the domain
- `call_dom(dom_id)`
- if metadata says `HC_WRITE_STDOUT`, perform Linux `write(1, payload + offset, length)`
- store the result back into metadata
- `call_dom(dom_id)` again
- verify the domain exits cleanly

### Domain side

- copy `"hello from domain\n"` into payload region
- fill metadata with `HC_WRITE_STDOUT`
- return to host
- on second entry, read `result/error`
- terminate cleanly

### Host harness pseudocode (v0)

```c
capstone_init();

dom_id = create_dom("/test-domains/sbi.dom", "/test-domains/hostcall_puts.smode");

metadata_id = create_region(4096);
payload_id  = create_region(4096);

metadata = map_region(metadata_id, 4096);
payload  = map_region(payload_id, 4096);

zero(metadata, 4096);
zero(payload, 4096);

shared_region_annotated(dom_id, metadata_id,
    CAPSTONE_ANNOTATION_PERM_INOUT,
    CAPSTONE_ANNOTATION_REV_SHARED);

shared_region_annotated(dom_id, payload_id,
    CAPSTONE_ANNOTATION_PERM_INOUT,
    CAPSTONE_ANNOTATION_REV_SHARED);

rv = call_dom(dom_id);

assert(rv == HC_V0_RET_PENDING);
assert(metadata->phase == HC_V0_PHASE_REQ);
assert(metadata->opcode == HC_V0_OP_WRITE_STDOUT);
assert(metadata->offset + metadata->length <= 4096);

host_rc = write(1, payload + metadata->offset, metadata->length);

metadata->result = host_rc;
metadata->error = (host_rc < 0) ? errno : 0;
metadata->phase = (host_rc < 0) ? HC_V0_PHASE_ERROR : HC_V0_PHASE_RESP;

rv = call_dom(dom_id);

assert(rv == HC_V0_RET_DONE);
assert(metadata->phase == HC_V0_PHASE_DONE);
assert(metadata->result == expected_len);

capstone_cleanup();
```

### Domain-side pseudocode (v0)

The domain-side logic should behave like a tiny two-state machine.

```c
static struct hostcall_v0 *meta;
static char *payload;
static int stage;

main() {
    if (!stage) {
        region_n = REGION_COUNT();
        meta = REGION_BASE(region_n - 2);
        payload = REGION_BASE(region_n - 1);

        copy(payload, "hello from domain\n", 18);

        meta->opcode = HC_V0_OP_WRITE_STDOUT;
        meta->offset = 0;
        meta->length = 18;
        meta->result = 0;
        meta->error = 0;
        meta->phase = HC_V0_PHASE_REQ;

        stage = 1;
        DOM_RETURN(HC_V0_RET_PENDING);
    }

    if (stage == 1) {
        if (meta->phase != HC_V0_PHASE_RESP || meta->error != 0) {
            meta->phase = HC_V0_PHASE_ERROR;
            DOM_RETURN(HC_V0_RET_ERROR);
        }

        if (meta->result != 18) {
            meta->phase = HC_V0_PHASE_ERROR;
            DOM_RETURN(HC_V0_RET_ERROR);
        }

        meta->phase = HC_V0_PHASE_DONE;
        stage = 2;
        DOM_RETURN(HC_V0_RET_DONE);
    }

    meta->phase = HC_V0_PHASE_ERROR;
    DOM_RETURN(HC_V0_RET_ERROR);
}
```

Important note:
- the first experiment should avoid libc entirely on the domain side
- use a tiny local copy routine instead of `memcpy`
- keep the message length fixed and explicit to reduce moving parts

### Success criteria for the first experiment

The first experiment should count as successful only if **all** of the following
are observed:

1. Host successfully creates the domain and both regions.
2. First `call_dom()` returns `HC_V0_RET_PENDING`.
3. Shared metadata after the first call is internally consistent:
   - `phase == HC_V0_PHASE_REQ`
   - `opcode == HC_V0_OP_WRITE_STDOUT`
   - `offset == 0`
   - `length == expected_len`
4. Host prints the exact payload bytes to stdout exactly once.
5. Host writes response data back into metadata and performs a second `call_dom()`.
6. Second `call_dom()` returns `HC_V0_RET_DONE`.
7. Metadata ends in `HC_V0_PHASE_DONE` with `result == expected_len` and `error == 0`.
8. No QEMU assert, no kernel-module failure, no unexpected capability fault.

### What should *not* be part of v0

To keep the first proof maximally narrow, do **not** add any of the following yet:
- general `write(fd, ...)`
- host-visible register ABI for arguments
- resumable mid-call trap handling
- dynamic allocation inside the domain
- libc integration
- bidirectional buffer ownership gymnastics beyond the shared-memory protocol above

## Why this is the right near-term milestone

This milestone:
- proves the real host-service architecture using already existing primitives
- avoids blocking on libc/sysroot work too early
- avoids assuming a resume ABI that is not yet clearly exposed to userspace
- creates the foundation for later `puts`, `write`, and small micro-libc steps

## What to postpone until after this proof

- full hosted glibc/sysroot compatibility work
- `picolibc` / `newlib` porting
- speculative yield/resume ABI work
- larger application bring-up such as sqlite/libpng/FFmpeg

## Historical runtime caveat and its current resolution

During the first implementation attempt, the draft HostCall probe appeared to show
that shared-region mutations were not becoming visible back in the host helper and
that the split `null_blk` reference path was crashing.

That caveat is now understood as a **historical local-environment issue**, not the
current baseline architecture result.

The decisive later finding was that the local tree had lost:

- `capstone/caplifive-buildroot/build/local.mk`

Without that override file, the local image fell back to a stock OpenSBI path, so
host-side `SBI_EXT_CAPSTONE` calls behaved like the wrong firmware image rather than
like the intended Capstone-enabled runtime.

After restoring `build/local.mk`, rerunning the validated OpenSBI rebuild path, and
rebuilding `capstone-null-blk` so its `vermagic` matched the active kernel, the
following were all revalidated:

1. the shared-region probe now observes host-visible sentinel changes across two
   successive `call_dom()` rounds,
2. baseline `null_blk` works,
3. split `null_blk` now creates `/dev/nullb0`, completes I/O, and unloads cleanly.

### What this means now

The architectural direction remains the same:
- split host-enclave
- shared-memory RPC
- host-side service execution

But the immediate coding milestone is no longer “prove that shared-region plumbing
works at all.” That part now has a validated baseline again.

### Refined next micro-step

The new smallest meaningful implementation step is to move **up one layer** from the
sentinel proof toward the first real HostCall proof-of-concept, for example:

1. keep using the restored shared-region path as the runtime baseline,
2. implement the narrowest real request/response experiment on top of it (such as the
   previously outlined `HC_V0_OP_WRITE_STDOUT` flow),
3. validate that higher-level proof with the same style of two-round `call_dom()`
   runtime harness,
4. only then generalize the ABI or broaden the hosted-software ambition.

So the gating question is no longer “is shared-region mutation broken in the current
tree?”. It is now “what is the smallest real host-service protocol we should prove on
top of the now-working runtime baseline?”.

