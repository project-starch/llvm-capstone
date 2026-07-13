# Borrow-path cost — measurement results (paper deliverable 2, task-014)

**Claim under test (paper conclusion, deliverable 2):** *"a measurement showing
the capability-mediated borrow path stays close to raw pointers and well below
the copy baseline."*

**Vehicle: (a) dynamic-instruction-count proxy on the QEMU functional model.**
The human/PI selected vehicle (a) for the gated step. See the methodology caveat
below — this is an honest overhead proxy, **not** cycle-accurate silicon timing.

## Setup

One domain (`borrow_cost.c`, Capstone clang, `-O2`) receives a real
monitor-granted **LINEAR** arena (the same `REGION_SHARE` / `REV_TRANSFERRED`
delivery the intra-domain MREV/REVOKE validation uses) and runs three variants
of the **same** boundary operation — *borrow one result word across the
host/engine boundary and use it* — differing only in how the lend is protected:

| Variant | What it does per operation |
|---|---|
| **raw pointer** | today's zero-copy path: dereference the shared word (no capability machinery). |
| **capability borrow** | the paper's mechanism: mint a revocation cap (`mrev`) + delegate a working cap (`delin`) + access it + `revoke`; the reclaimed LINEAR handle is threaded into the next lend. |
| **copy baseline** | the `TRANSIENT`-style defensive copy the mechanism replaces: `memcpy` the payload into a private buffer, then read it. |

Each variant runs a 1024-iteration inner loop bracketed by **`csrdicount`**, a
new emulator readout that returns QEMU's raw retired-instruction count (icount).
The boot adds `-icount shift=0`, so one icount tick == one retired instruction
and a `csrdicount` delta is an exact dynamic instruction count. An empty
calibration loop of the same shape is subtracted to isolate the operation from
the shared loop-control + bracket overhead:

    instr/op = (variant_total − empty_loop_total) / iterations

Reproduce: `capstone/tests/runtime-qemu/run-borrow-cost-probe.sh` (needs the
rootfs lock; one QEMU boot).

## Results

Dynamic instruction count per boundary operation (`-O2`, 1024 iterations,
deterministic — the counts are exact integers and reproduce bit-for-bit):

| Variant | instr / op | vs raw | Cost model |
|---|---:|---:|---|
| raw pointer          |   **2** | 1.0× | O(1) |
| **capability borrow**|   **6** | **3.0×** | **O(1), payload-independent** |
| copy — 256 B payload |    34 | 17.0× | O(payload) |
| copy — 1024 B payload |  130 | 64.9× | O(payload) |

- **Close to raw pointers.** The borrow path costs **4 extra instructions** over
  a raw pointer load — one each for `mrev`, `delin`, `revoke`, and threading the
  reclaimed handle — a small **payload-independent constant** (6 vs 2 instr,
  3.0×). The four capability ops *are* the mechanism the paper describes.
- **Well below the copy baseline.** Borrow is **5.7× cheaper** than even a 256 B
  defensive copy and **21.7× cheaper** at 1024 B, and the gap **widens without
  bound**: the copy is `2 + payload_bytes/8` instructions (**O(payload)** — 34 at
  256 B, 130 at 1024 B, an exact fit) while borrow and raw are **O(1)**.

### Static cross-check (exact)

The dynamic counts match the `-O2` disassembly loop bodies exactly, as expected
for straight-line bodies on a functional model (`asm/borrow_cost.s`):

```
empty:  addi; bnez                                              = 2 instr
raw:    ld; addi; xor; bnez                                     = 4 instr (marginal 2: ld + use)
borrow: mrev; delin; ld; revoke; addi; xor; movc; bnez         = 8 instr (marginal 6)
copy:   16-byte ldc/stc pairs over the payload + use           = 2 + payload/8
```

The borrow marginal (6) is `mrev`, `delin`, the delegated load, `revoke`, the
reclaimed-handle `movc`, and the `xor` that uses the value.

## Methodology caveat (read before citing)

**This is a functional-model instruction-count proxy, NOT cycle-accurate
timing.** QEMU is an ISA/functional emulator: it executes the right instructions
and faults but models **no** microarchitecture — no pipeline, cache, or cycle
model. `csrdicount` therefore measures **dynamic instructions retired**, which is
an honest, deterministic overhead proxy but does **not** account for per-
instruction latency, memory-hierarchy effects, or ILP. A true "stays close to
raw pointers" *timing* claim needs a cycle-accurate vehicle (Capstone RTL on an
FPGA, or a gem5-class model) that is not in this tree.

Any number taken from here into the paper must be labelled as a
**functional-model instruction-count proxy**. What the proxy *does* establish
rigorously and is safe to state: the borrow mechanism adds a **small constant
number of instructions** (four capability ops) over a raw pointer, **independent
of payload size**, whereas the defensive copy it replaces is **linear in the
payload** — so the borrow path is close to raw and its advantage over copying
grows with the size of the borrowed value.

## Scope

Additive. The measurement needed one emulator feature — the `csrdicount`
instruction-count readout (`capstone/capstone-qemu`, gitlink bumped, its own
lane) — plus a new probe under `tests/runtime-qemu/` and an additive
`--qemu-extra-arg` passthrough in `run-domain-smoke.py`. No `llvm/` change; A's
row repros, `start.S`, the monitor, the shared allocators, and `capstone-c` are
untouched.
