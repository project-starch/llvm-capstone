# Virtual C: service round trips without the module

Status: **plan, not started.** Nothing below has been built or measured.
Branch `plan/user-mode-resume` (llvm-capstone), from `dev` 329547f1e0a3 with
QEMU pin 1a6dd2073206.

## Goal and constraints

Linux is in the TCB of the supervised virtual C runtime, yet every service
call of an application still passes through `capstone_vm.ko`. This plan
removes the module from the ordinary service path.

- **No Linux patch.** Linux core, its entry code and its signal handling stay
  as they are, and the design must not depend on their internals.
- **Prefer the processor.** Where a processor primitive makes the software
  simpler or safer, add it. Everything else goes into the kernel module.
- **Every processor primitive must map to RTL.** Per hart that means a few
  CSRs, fixed memory layouts and simple checks. QEMU-only conveniences do not
  count as design. The [primitive summary](#processor-primitives-summary)
  lists everything the plan introduces.
- **One version.** The virtual `CSRUNV` ABI is replaced outright. There are no
  compatibility paths and no feature probes.

## Today: one ioctl round trip per service

The launcher worker drives its context through `CV_STEP`
(`runtime/virtual/exec.c:588`). The module runs the context with `CSRUNV`
under `vm_lock`, with interrupts and preemption disabled
(`runtime/virtual/module/capstone_vm.c:781`).

```
 App (virt. C)     Launcher worker (U)        Linux (S)              Module (S)
   ECALL ─────────────────────────────────────────────────(1)──▶ CSRUNV returns
                   CV_STEP returns ◀───────────────────────(2)─── (kind 3)
                   serve request
                   ecall (e.g. write) ──(3)─▶ sys_write
                                  ◀──(4)─── sret
                   CV_STEP(reply) ────────────────────────(5)──▶ CSRUNV action 2
   resumes after ◀────────────────────────────────────────(6)───
   ECALL
```

Six transitions and two kernel round trips per service. A service that makes
no Linux call still pays the ioctl.

## Target: the worker resumes its own context

```
 App (virt. C)     Launcher worker (U)        Linux (S)              Module (S)
   ECALL ──(1)───▶ CSRUNV returns                                 (not involved)
                   (kind 3, U→U in the processor)
                   serve request
                   ecall (e.g. write) ──(2)─▶ sys_write
                                  ◀──(3)─── sret
                   CSRUNV (reply)
   resumes after ◀──(4)─ (U→U in the processor)
   ECALL
```

Four transitions and one kernel round trip, which is the service's own
syscall. A service without a Linux call needs no kernel entry at all.

## What the QEMU prototype does today, and why it does not map to RTL

`target/riscv/capstone_supervisor.c` keeps every continuation in a hidden
table of 32 slots per hart. Each slot holds a full copy of the CPU state,
including every machine CSR, `mip`/`mie` and a vector file. The caller's
state is snapshotted the same way. A 5 ms virtual-clock quantum timer forces
an escape, and interrupts stay masked while the application runs.

On silicon this would mean tens of KiB of hidden state per hart and an extra
timer. It also has a defect: restoring the snapshot's `mip` can overwrite an
interrupt raised during the run. The Capstone ISA already has an RTL-shaped
model for this: an interrupted context is saved to memory
(academic spec, `parts/int-except.adoc`, "the current context is saved and
sealed"). This plan moves the virtual path to that model.

## Processor primitives

### P1. Context records in a kernel-owned region

Two new S-mode CSRs describe one physically contiguous, 4-KiB-aligned
region that the module allocates at load time:

| CSR | Access | Meaning |
|---|---|---|
| `vcbase` | S read/write | Physical base of the context-record region |
| `vccount` | S read/write | Number of 4-KiB records in it; zero disables virtual C |

Written once by the module at load. A context is named by its **record
index**, and the processor computes `vcbase + index * 4096`. Records are
ordinary kernel memory that U mode cannot reach. Every field below is
therefore trusted, and no physical address is exposed to user space.

Record layout (4 KiB, little-endian):

| Offset | Size | Field | Written by |
|---|---|---|---|
| 0 | 8 | State: 0 free, 1 ready, 2 running, 3 paused, 4 terminal; bits 8..15 last event kind | processor (module: free→ready) |
| 8, 16, 24 | 8 each | Event cause, PC, fault address | processor |
| 32 | 8 | Binding: `satp` MODE and root PPN of the owning `mm` | module |
| 40 | 8 | Physical lifetime-table root (`srevroot`) | module |
| 48 | 8 | Start-frame physical address (first entry only) | module |
| 56 | 8 | Options (resumable ECALL, resumable page faults) | module |
| 64 | 16 | Tagged reply slot (S-mode replies only) | module |
| 80 | 64 | Scalar views of the application's `a0..a7` at the last event | processor |
| 144 | 16 | Collection request: page count, page-list address | module |
| 256 | 512 | Application PCC and `x1..x31`, 16 bytes each plus tag | processor |
| 768 | 264 | Application `f0..f31` and `fcsr` | processor |
| 1040 | 24 | Host PC, privilege, `srevroot` | processor |
| 1072 | 496 | Host `x1..x31`, 16 bytes each plus tag | processor |
| 1568 | 264 | Host `f0..f31` and `fcsr` | processor |

The save and restore of the two register files is the same kind of sequenced
load/store work as the ISA's existing context save on interrupts. The hart
keeps only the index of the running record (P4). This replaces the 32 hidden
slots, the caller snapshot and the `mip` restore.

### P2. Namespace control word

The paged lifetime-table root has two reserved zero words at offsets 48 and
56. Word 48 becomes the namespace control word:

| Bits | Field |
|---|---|
| 63 | HOLD |
| 31..0 | RUNNING: number of contexts of this namespace currently executing |

Every entry into a context increments RUNNING with one atomic update, and
every escape decrements it. A **U-mode** entry fails if HOLD is set, checked
in the same atomic update. S-mode entries ignore HOLD: they are the module's
own entries, serialized by `vm_lock`, so the holder can still run contexts
(for example `CV_STEP` after a fault). The module holds a namespace by setting HOLD with an AMO,
then waits until RUNNING is zero. On one hart RUNNING is already zero
whenever the module runs. On several harts the module sends an IPI to force
escapes, so the protocol stays correct without changes.

This replaces the `vm_lock` exclusion that a U-mode resume would bypass
(details under the module changes). Virtual C requires the paged table
format. The flat format has no header space and is dropped for virtual C.

### P3. `CSRUNV`: same encoding, defined by privilege

`CSRUNV rd, rs1, rs2` (opcode `0x5b`, funct3 `1`, funct7 `0x24`), with `rs1`
holding the record index.

**S mode**: `rs2` is the action:

| Action | Precondition | Effect |
|---|---|---|
| 0 Start/resume | ready, or paused at kind 1 or 4 | Ready: consume the start frame's initial slots (as today) into the record and start. Paused: resume at the saved PC |
| 1 Forget | not running | State → free; registers in the record cleared |
| 2 Reply | paused at kind 3 | Move the tagged reply slot into `a0`, clear the slot, PC + 4, resume |
| 3 Collect | paused | Collection over the namespace, as today |

**U mode**: `rs2` is a scalar reply. The processor checks, in this order:

1. `index < vccount` and the record is paused;
2. the binding equals the caller's current `satp` MODE and root PPN;
3. the last event kind is 1 (interrupt) or 3 (service);
4. HOLD is clear (checked atomically with the RUNNING increment, P2).

If checks 1-3 fail, `rd` = 7 (refused); if check 4 fails, `rd` = 6 (held).
Nothing runs and there is no trap. A refusal in U must not raise a
capability cause, because Linux would deliver it to the trusted launcher as
an unknown exception. Checks 1 and 2 give the same answer for a foreign or a
nonexistent record. On success, kind 3 writes the reply as untagged `a0` and
advances the PC by four; kind 1 resumes at the saved PC.

U mode can never start, forget, collect or install a tagged value. Virtual C
rejects `CSRUNV` in every form.

Entry, in both privileges, also requires the binding to equal the current
`satp`. The module's ioctls already run in the owning `mm`. Entry then saves
the host state (P1), loads the application state, sets the state to running
and the running index (P4), selects the record's `srevroot`, and switches to
virtual C.

No other CSR is saved or swapped. `satp` stays, because the binding equals
it. Interrupt, trap and status CSRs stay too, because virtual C can neither
read nor write CSRs. The application therefore runs under the host's
interrupt configuration, and that is what P4 relies on.

### P4. Running index and escape

One per-hart CSR, read-only to software:

| CSR | Meaning |
|---|---|
| `vcactive` | Index + 1 of the running record, zero when no virtual context runs |

An escape saves the application state into the record, writes the event, sets
the state to paused (terminal for kind 2), decrements RUNNING and restores the
host from the record. It then writes `rd`:

| Kind | Cause of the escape |
|---|---|
| 1 | An interrupt is pending that the host's privilege or M mode would take, ignoring the host's global enable (`sstatus.SIE`) |
| 2 | Terminal fault, including any exception not listed here |
| 3 | Service ECALL |
| 4 | Resumable page fault |
| 5 | Node pressure |

For a U host and kind 3, the processor also copies the scalar `a0..a7` views
into the host's `x10..x17`. `rd` must not be one of them; the launcher uses
`t0`.

M-mode interrupts escape too. Otherwise firmware would handle them while
the application's capability registers are live, and its scalar register
save would lose the tags. Kind 1 replaces the quantum timer. Linux's own timer tick and device
interrupts end a run: the escape happens first, and then the hart takes the
interrupt in the host, either at once (U host) or when the module re-enables
interrupts (S host). Preemption becomes Linux's scheduling policy, and the
processor holds no timer and no interrupt state of its own.

### Processor primitives summary

| Primitive | RTL state | RTL logic |
|---|---|---|
| P1 records | 2 S-mode CSRs (`vcbase`, `vccount`) | index → address, range check; sequenced save/restore of two register files |
| P2 control word | none on the hart (memory word) | one AMO on entry, one on escape |
| P3 `CSRUNV` from U | none | 4 checks before entry |
| P4 escape | 1 CSR (`vcactive`) | interrupt-pending test against the host privilege; event write; `x10..x17` copy for U hosts |
| Removed | 32-slot table, caller snapshot, quantum timer, `mip`/`mie` snapshot | |

Not added: no TLB tag. On silicon the application and its worker share one
`satp`, so the translations are identical. QEMU flushes its soft TLB on every
switch only because virtual C uses a different internal MMU index. Giving
virtual C its own index is a QEMU performance change, not an ISA change.

## Module changes (`runtime/virtual/module/capstone_vm.c`)

1. **Record region.** Allocate it at module load, write `vcbase`/`vccount`,
   and allocate records per context. Main-thread start frames stay
   kernel-owned. Child start frames stay in registered user pages, but the
   processor reads them only at first entry. All later state lives in the
   record.
2. **Events from the record.** The module reads kind, cause, PC and fault
   address from the record, which only the processor writes. There is no
   separate `t->event` bookkeeping that U resumes could make stale, and no
   application-writable frame in the loop.
3. **Hold for every ioctl.** After taking `vm_lock`, hold the namespace (P2)
   and release it before unlocking. This keeps today's guarantee that no
   context runs while the module sleeps. That guarantee matters for
   node-page publication (`node_add_page`, `:79`), the pin scan before
   collection (`collect`, `:315`), `add`, and the remap transaction between
   `CV_REMAP_BEGIN` and `CV_REMAP_END`, during which the hold stays set. A
   second worker resuming in that window would otherwise run C code. For
   example, it could store a capability into a resident page that the
   collection scan has already passed, before IDs are recycled.
4. **Handles.** `CV_STEP` and `CV_THREAD_CREATE` return the record index.
5. **`CV_EVENT`** (new): return kind/cause/pc/address/args from the record
   without running. Used for kinds 2, 4 and 5 received in U.
6. **`CV_STEP` after kind 5** calls `ensure_nodes` before resuming.
7. **Statistics.** `steps` counts module-run steps; the launcher counts U
   resumes. Both are printed in `CAPSTONE_VM_STATS`.

Unchanged module duties: start, mint/retire, mapping replies, fault resolution
and pinning, node growth and collection, thread lifecycle and teardown.

## Launcher changes (`runtime/virtual/exec.c`)

`virtual_service_loop` (`exec.c:588`) becomes:

```
first: CV_STEP (module start)                      -> kind, index
loop:
  kind 1 -> kind = csrunv_u(index, 0)              (Linux already took the interrupt)
  kind 3 -> serve (virtual_service, unchanged)
            reply is a capability (MAP, REMAP) -> CV_STEP(reply)
            otherwise                          -> kind = csrunv_u(index, result)
  kind 6 -> CV_STEP(reply if pending)   (held: the module resumes after its work)
  kind 2, 4, 5 -> CV_EVENT; then fault exit / CV_RESOLVE + CV_STEP / CV_STEP
  kind 7 -> die("resume refused")       (a launcher bug, never expected)
```

`csrunv_u` is an inline-asm wrapper: index in `rs1`, reply in `rs2`, kind in
`t0`, `a0..a7` clobbered and returned as the request arguments. The quantum
`sched_yield` goes away, because Linux preempts at its own tick. The
`service_lock` and the per-worker exchange and signal state are unchanged.

## Unchanged

The application, the capability libc and its exchange-region bridge, the
delegated wire format, Linux, firmware and the trust boundary. Linux still
touches only the exchange region, through the worker.

## Safety argument (to be checked in review)

- **TCB unchanged.** The worker could already step any context of its `mm`
  through the module; U resume only shortens that path.
- **Records cannot be forged or named from outside.** They live in a
  kernel-only region and are named by index. The binding check rejects every
  record of another `mm` with the same code as a nonexistent one.
- **No authority injection.** U writes only an untagged `a0`. Tagged replies,
  start, forget and minting stay S-only.
- **No resume past an unresolved fault or pressure.** U refuses kinds 2, 4
  and 5, so a page is pinned before the faulting access retries.
- **Exclusion enforced by the processor.** HOLD and RUNNING give every module
  operation the "no C execution" guarantee of `vm_lock`, independent of
  launcher discipline, and on any number of harts.
- **No lost interrupts.** The processor neither masks nor snapshots interrupt
  state. A pending interrupt ends the run and is then taken normally.

## Open questions

1. **Binding and ASIDs.** The binding stores MODE and root PPN only. Linux can
   reassign an `mm`'s ASID, but not its root page table.
2. **Region size.** One region for all processes; at 4 KiB per context, 1024
   contexts need 4 MiB. The size could be a module parameter. Running out is
   reported to the caller as `-ENOSPC`, as the 32-slot limit is today.
3. **Switch cost.** The record is 1.8 KiB, but one escape moves less. It
   stores the application's PCC and registers (512 B) and FP state (264 B),
   and loads the host's registers (496 B), FP state (264 B) and three words.
   That is 776 B stored and 784 B loaded in 132 memory operations; an entry
   is the mirror image. A Linux trap entry saves about 35 eight-byte words.
   Cycles depend on the core's store path and on caching and need an RTL
   estimate. The reductions are deferred until the path runs (see Later).
4. **Physical path.** `CSUPERVISE`/CALL keep the slot table. Moving them to
   records is a separate change.

## Tests and gates

Runs go to p12/p13, not the local host. Scope and expected duration are
stated before each run.

Existing gates must pass on the new pin: `run.py`, `run-pthreads.py`,
`run-malloc.py`, `run-nodes.py`, `run-safety.py`, `run-ports.py`, the M1 and
U-access processor suites, and the signal and socket contracts.

Processor tests (bare-metal probes beside the M1 suite):

| Test | Expectation |
|---|---|
| T1 U resume after kind 1 | runs; returns the next event kind |
| T2 U resume with reply after kind 3 | `a0` = reply, tag clear, PC + 4 |
| T3 U resume after kinds 2, 4, 5; of a free, ready or running record; index ≥ `vccount` | 7, no change to the record |
| T4 U resume with another `satp` root | 7 |
| T5 HOLD set | U gets 6; S actions 0 and 2 still run; RUNNING counts both |
| T6 actions 0, 1, 3 and tagged replies from U; any `CSRUNV` from virtual C | 7 / illegal instruction as specified |
| T7 interrupt raised during a run | escape kind 1, interrupt taken in the host, nothing lost |
| T8 escape and resume preserve host and application registers | full-width registers and tags compare equal |
| T9 RUNNING returns to zero after every escape kind | control word checked after each |

Runtime tests:

| Test | Expectation |
|---|---|
| R1 a service storm while another thread grows nodes and collects | no ID recycled under a live tag |
| R2 mremap while another thread runs | that thread gets 6 and continues after `CV_REMAP_END` |
| R3 signal to a worker during a long C loop | delivered after the next tick; signal contract 30/30 |
| R4 another process guesses indices | 7 for all |

## Measurement

Same images and SDK before and after; only QEMU, module and launcher change.

- **Structural counters (deterministic):** module steps and U resumes per
  run, per service kind. Expected: module steps drop to starts + faults +
  capability replies + pressure + held retries.
- **Time:** a contract mode with N delegated `getppid` calls and N local
  `THREAD_SELF` calls. Report launcher elapsed time per call over repeated
  runs. These are QEMU numbers only, not RTL claims.
- **Applications:** SQLite speedtest1 and the mruby gate: elapsed time, and
  the share of services that still reach the module.

## Work breakdown and size

Sizes are estimates from the current files: `capstone_supervisor.c` 957
lines, of which the virtual path is about 290; the module 955; `exec.c` 736.

| # | Repository / branch | Content | Estimated change |
|---|---|---|---|
| 1 | capstone-qemu, branch from 1a6dd2073206 | Records and save/restore, `vcbase`/`vccount`/`vcactive`, control word, U-mode `CSRUNV`, interrupt escape; the virtual path stops using slots and the quantum timer; the physical slot path stays | ~450 lines changed or new |
| 2 | same | Bare-metal probes T1-T9 beside the M1 suite | ~600-800 lines |
| 3 | llvm-capstone, stacked on this plan | Module 1-7, `wire.h` | ~250 lines |
| 4 | same | Launcher loop, `csrunv_u`, statistics | ~120 lines |
| 5 | same | R1-R4, contract timing mode; QEMU pin bump | ~250 lines |
| 6 | same | `design/virtual-capstone/isa.md`, `runtime.md`; academic-spec `virtual-capstone.adoc` | ~250 lines of prose |
| 7 | p12/p13 | Existing gates, then the measurement | runs only |

About 1,000 lines of implementation and 1,000 of tests and documentation, as
two commits in capstone-qemu (1+2), one llvm-capstone lane (3-6) and one
upstream PR for the QEMU change before the pin bump.

## Later

- **Cheaper switches** (after the path runs and is measured):
  - Declare caller-saved host registers clobbered by `CSRUNV`, like a
    function call. The processor then saves only `ra`, `sp`, `gp`, `tp`,
    `s0..s11`, `fs0..fs11` and `fcsr`, scalar. The host side drops from
    784 B to 256 B.
  - Save application FP only when `mstatus.FS` is dirty. The application
    side drops from 776 B to 512 B in integer code.
  - Together: 64 instead of 132 memory operations per escape.
  - A second register bank (about 4 Kbit per hart) would make one context
    nearly free to switch, but several threads per hart still need records.
    Not planned.
- **Stage 2:** `CSMINT`/`CSRETIRE` from U, using the same binding check, if
  the measurement shows that mapping services matter. Pinning and fault
  resolution stay in the module.
- **Not planned:** sending application ECALLs directly into Linux's trap
  vector. It would depend on Linux entry, restart and signal-frame internals,
  which the no-Linux-patch constraint excludes in spirit even without a
  source patch.
