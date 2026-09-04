# How "system calls" work in the Capstone domain model 

## TL;DR

There are **no classic system calls** (`ecall`/trap-to-kernel) in the path we run.
A Capstone **domain** is a freestanding, bare-metal, capability-isolated unit.
When it needs a host service it uses **HostCall v0**: a *cooperative,
shared-memory request/response protocol* where the shared buffers are passed as
**capabilities with explicit permission and revocation annotations**, and control
is transferred by a **capability domain-call/return**, not by a trap. The three
benchmark suites (CoreMark, BEEBS, RV8) use **none of this** — they are pure
computation that returns a result via a memory marker.

---

## 1. What a "domain" is

A domain is an ELF-like image loaded at a runtime base, run under a
Capstone-enabled OpenSBI/monitor, with a small set of **root capabilities**
(`gp` = global/code/data root, `sp` = stack root) handed to it at entry. It does
**not** have an ambient OS, libc, or syscall table. Memory is whatever the loader
gives it; isolation between domains is enforced by the monitor bounding each
domain's roots (plus page tables/PMP).

---

## 2. Domain entry and exit (the transition primitives)

From `start.S`, the relevant custom instructions (opcode `0x5b`):

| Instr | Encoding | Role |
|-------|----------|------|
| `ccsrrw rd, ccsr, rs` | `.insn i 0x5b,0x7` | read/write a **capability CSR** (e.g. `cscratch`) — used to fetch/stash the boot/scratch capabilities at entry and exit |
| `scc` / `lcc` | `0x5b,0x1,0x5` / `0x1,0x4` | set cursor / query a capability field (tag, cursor, base, end, perms) |
| `delin` | `0x5b,0x1,0x3` | de-linearise a (linear) capability so it can be used freely |
| `domreturn rd, rs1, rs2` | `.insn r 0x5b,0x1,0x21` | **return control out of the domain** to the monitor/host |

**Entry (`_start` → `test` → `domain_main`):** recover `sp`/`gp` from the boot
capabilities via `ccsrrw`, set up the stack, **run the capability-global
initializers** (the `.capstone_cap_init` PC-relative table — see
`capability-globals-init-decision.md`), then call the C entry point
`domain_main`.

**Exit:** after `domain_main` returns, `start.S` **scrubs every general register
to zero**, stashes `gp`/scratch back through `cscratch`, clears `mcause`/`mtval`,
and executes **`domreturn`** to hand control back. The register scrub is a
deliberate **capability-leak hygiene** step: the domain must not return with stray
capabilities live in registers. This is the closest thing to "return from a
syscall," but it is a *capability domain boundary crossing*, not a kernel trap.

---

## 3. HostCall v0 — the actual "syscall" mechanism

HostCall is how a domain asks the outside world to do something it cannot
(write to a real fd, touch a file). It is a **two-party state machine over shared
memory**, not a trap.

### 3.1 Shared regions are capabilities with annotations

Before the conversation, the host grants the domain access to one or more shared
memory regions via `shared_region_annotated(dom_id, region_id, perm, rev)`:

- **permission**: `PERM_IN` (host→domain), `PERM_OUT` (domain→host), `PERM_INOUT`.
- **revocation/lifetime**: `REV_BORROWED` (single round, revoked after) or
  `REV_SHARED` (stays live across rounds).

Typically: a small **metadata** region stays `INOUT + SHARED`, and a larger
**payload** region is `OUT + BORROWED` (lent for exactly one round). This is the
capability analogue of `copyin`/`copyout` — except instead of copying through a
trusted kernel, the host is *handed a bounded capability* to exactly the bytes it
may touch, for exactly as long as it may touch them.

### 3.2 The metadata block (`struct hostcall_v0`)

```c
struct hostcall_v0 {
  u64 phase;    // INIT(0) -> REQ(1) -> RESP(2) -> DONE(3) / ERROR(4)
  u64 opcode;   // which service (table below)
  u64 offset;   // byte range inside the payload region the host should consume
  u64 length;
  s64 result;   // host-written result
  s64 error;    // host-written errno-like code
};
```

### 3.3 The protocol (round-trip)

1. Domain fills `metadata` (sets `opcode`, `offset`, `length`, `phase = REQ`) and
   writes request bytes into the payload region.
2. Domain **yields** (`domreturn`) → returns a scalar status to the host:
   `PENDING(1)` means "I need servicing."
3. Host helper (`call_dom` re-enters the domain after servicing) **snapshots** the
   request fields *immediately* (it does **not** trust repeated reads of mutable
   shared state), performs the real OS action (e.g. `write(STDOUT_FILENO, …)`),
   writes `result`/`error`, sets `phase = RESP`, and re-enters the domain.
4. Domain reads the response, sets `phase = DONE`, returns `DONE(0)`.

For multi-step services (file handles) the same metadata ABI is reused across
several REQ/RESP rounds, with an explicit **revoke-before-reborrow** of the
borrowed payload between rounds.

### 3.4 Service surface (opcodes implemented today)

```
1  WRITE_STDOUT            16 FILE_OPEN        20 FILE_STAT_BASIC   23 PATH_ACCESS
2  WRITE_GUEST_TMPFILE     17 FILE_READ        21 FILE_SYNC         24 PATH_DELETE
3  READ_GUEST_TMPFILE      18 FILE_WRITE       22 FILE_TRUNCATE
4-7 SECOND_PENDING (multi-round test shapes)   19 FILE_CLOSE
```

i.e. an embryonic OS ABI: stdout + a handle-based file API + path operations.
Validated end-to-end flows include `OPEN→WRITE→SYNC→CLOSE→OPEN→READ→CLOSE`. The
**SQLite VFS skeleton** (README item 20) is the first "real software" consumer —
its file I/O is intended to route through these HostCall file ops.

---

## 4. Why this design (security framing)

- **No ambient authority.** A domain can only reach a service if it was *handed a
  capability* to the shared region for it. There is no global syscall table to
  reach; absence of a grant = absence of the capability.
- **Bounded, time-scoped sharing.** Borrowed payloads are bounded capabilities,
  revoked after one round, so the host's exposure to domain memory (and vice
  versa) is explicit and minimal — a capability-passing ABI rather than
  trust-the-kernel copying.
- **Leak hygiene at the boundary.** The domain scrubs all registers and stashes
  roots before `domreturn`, so crossing the boundary cannot leak a live
  capability.
- **Snapshot discipline.** The host copies request fields out of mutable shared
  memory before acting, preventing TOCTOU on the shared block.

---

## 5. What the benchmarks use 

CoreMark, BEEBS, and RV8 issue **no HostCall and no `ecall`**. They are compiled
freestanding (`-ffreestanding -fno-builtin`), use a local `memcpy`/string slice
and, where needed, a **static bump allocator** (no OS heap), run their kernel in
`domain_main`, and report success by writing a marker string
(`__…_PASSED__`) before returning. So the benchmark track and the syscall/HostCall
track are **independent**: HostCall is the experimental path toward hosted
services (and SQLite), exercised only by the `runtime-qemu` probe family.

