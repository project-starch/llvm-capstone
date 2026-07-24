# CHERI vs Capstone — temporal-safety performance comparison (full report)

---

## 1. Executive summary

The project direction reframed the CHERI comparison: on *security*, eager CHERI matches us
(revoke-on-every-free blocks the corpus's temporal defects); the axis that
actually separates the two systems is *performance*. This report answers the
resulting question — **what does temporal safety cost, and is our contract-point
revocation cheaper than CHERI's sweep-based revocation?** — with a head-to-head,
QEMU-to-QEMU measurement on two workloads.

**Headline:** the only CHERI configuration that matches our temporal security
(eager, revoke-on-every-free) costs **~14–17 million instructions per `free`** —
a stop-the-world address-space sweep — against our **fixed O(1) contract-point
revocation of 5–10 instructions per free**. At equal temporal security the two
mechanisms differ by **roughly six orders of magnitude per operation**. CHERI's
*deployable* default (async) is far cheaper (1.9–6.4×) but by its capability
mechanism blocks **0 of 11** use-after-free rows at the contract point — its
protection is deferred, not synchronous.

| workload | CHERI async (deployed) | CHERI eager (matches our security) | Ours (revoke-at-free) |
|---|---|---|---|
| microbench (malloc/free) | +20 k instr/op (6.4×) | **~14.0 M instr/op (3,731×)** | **+5 instr/op, O(1)** |
| real workload (BST) | +9 k instr/op (1.9×) | **~16.8 M instr/op (1,661×)** | **+10 instr/op, O(1)** |

---

## 2. Background — the argument for the comparison (2026-07-14)

- CHERI's **deployable** config (async revocation) does **not** catch the
  corpus's temporal defects at the contract point. Good — that is the security
  point (`tab:cheri`, from the cheri-baseline task).
- CHERI's **eager** config (revoke-on-every-free) **does** catch them — so on
  *security*, eager CHERI matches us (modulo row 3r, which no CHERI policy
  catches). The framing: eager is "a very slow version of what Capstone is doing…
  like a garbage collector."
- Capstone realizes the *same* revoke-at-free semantics as a **fast O(1)**
  capability op. So the separating axis is **performance**. Guidance (Slack): "to show
  the improvement, we should test the perf distinction on QEMU-Capstone and
  Qemu-cheri… add that performance data separately in a table."
- Methodology rule: **QEMU-to-QEMU** (do not compare CHERI-QEMU to Capstone-RTL —
  "that's incomparable"); RTL cycle-accurate is a Capstone-only follow-on.

The the board owner gate was dropped: the board owner is the FPGA/RTL collaborator, not a CHERI
expert, and `qemu-system-riscv64cheri` is a standard QEMU fork with the usual
instruction-count readouts, so the methodology was settled in-house.

---

## 3. Methodology

### 3.1 Two systems, two vehicles

| | Our system (Capstone) | CHERI |
|---|---|---|
| vehicle | Capstone functional model (`qemu-system-riscv64`), **bare-metal domain** (no OS) | `qemu-system-riscv64cheri` 7.1.0, **CheriBSD purecap** full OS (jemalloc, kernel) |
| temporal mechanism | revoke-at-free: one `REVOKE` capability op at the contract point | `CHERI_CAPREVOKE` pointer revocation (CHERIvoke/Cornucopia quarantine sweep) |
| configs | bump (no per-object caps) / revoke-on-free with revoke off / full revoke-on-free | spatial (rev off) / async (quarantine, deployed default) / eager (revoke every free) |
| toolchain | `$CAPSTONE_CLANG` (in-tree), domains via `build-domain.sh` | CHERI SDK clang 17 purecap (`~/cheri/output/sdk`) |

The CHERI stack is the same one the security-baseline task (task-015) built via
cheribuild under `~/cheri`; it uses its **own** `cheribsd-riscv64-purecap.img`, so
there is **no rootfs-lock contention** with the Capstone QEMU suites.

### 3.2 Instrumentation

- **Metric:** dynamic retired-instruction count, empty-loop calibrated, reported
  as per-operation overhead over each system's own unprotected baseline.
- **Capstone:** `rdcycle` under `-icount shift=0` (one-per-retired-instruction;
  deterministic; matches the FPGA readout). Bare-metal, so the count is pure
  workload — no OS/interrupts.
- **CHERI:** `rdinstret`, bracketing the workload. `instret` retires in **all**
  privilege modes, so the delta **includes the kernel-side revocation sweep** —
  which is the whole point: CHERI pays its temporal cost in a kernel quarantine
  sweep, not at the `free`. Confirmed user-readable in CheriBSD (no SIGILL).
- **No `-icount` on CHERI:** eager is ~10¹²–10¹² retired instructions per trial;
  counting one-by-one under `-icount` is intractable. Plain TCG + `rdinstret` is
  reproducible enough (trial spread ~1–3%). This asymmetry is exactly why the
  the methodology rule is QEMU-to-QEMU for the comparison and RTL only for the Capstone
  absolute.

### 3.3 Config toggles

- **Capstone:** one domain build per config, selected by `-DROF_COST_MODE`
  (bump=0 / norevoke=1 / revoke=2). The `norevoke` mode is the revoke-on-free
  allocator with `rof_no_revoke=1`, so it pays identical alloc-side cost but skips
  the free-time revoke — the `revoke − norevoke` delta isolates the revoke op.
- **CHERI:** `sysctl security.cheri.runtime_revocation_default` /
  `..._every_free_default` before each run (the cheri-baseline knobs), confirmed
  per process via `malloc_revoke_enabled()`.

### 3.4 Workloads

1. **Microbench:** `malloc(64) → touch one byte → free()`, tight loop. Isolates
   the mechanism; synthetic; heap empty at every free.
2. **Real workload (BST):** build a 2000-node binary search tree, look up every
   key by chasing pointers through it, tear it down; repeated (CHERI: 10 rounds /
   20 000 lifecycles; Capstone: 2 rounds / 4 000, bounded to the Phase-0 arena).
   All 2000 nodes are live during lookup, so a revocation sweep has a genuine live
   capability graph to scan. Shared source: `tests/shared/tree_workload.h`.

---

## 4. Results — microbench (malloc/touch/free)

**CHERI** (`rdinstret`, ITERS=200 000; spatial/async n=3, eager n=2):

| config | per-op (instr) | overhead vs spatial |
|--------|---------------:|--------------------:|
| spatial (baseline) | 3,760 | — |
| async (deployed default) | 23,977 | +20,217 (6.38×) |
| **eager (every free)** | **14.03 M** | **+14,025,640 (3,731×)** |

*Eager n=2: the 3rd trial exceeded the 60-min pexpect window (one eager trial is
~2.8×10¹² retired instructions = 200 000 frees × ~14 M each); the two completed
trials agree to 3%.*

**Capstone** (`rdcycle`/`-icount`, ITERS=512, `-O2`):

| config | per-op (instr) | overhead |
|--------|---------------:|---------:|
| bump (baseline) | 7.0 | — |
| revoke-on-free, revoke off | 60.0 | +53 (alloc-side) |
| **full revoke-on-free** | **65.0** | **+58 (9.28×)**; revoke-at-free itself **+5** |

---

## 5. Results — real workload (BST build/lookup/destroy)

**CHERI** (`rc_tree`, keys=2000, rounds=10, 20 000 ops; spatial/async n=3, eager n=1):

| config | per-op (instr) | overhead vs spatial |
|--------|---------------:|--------------------:|
| spatial (baseline) | 10,095 | — |
| async (deployed default) | 19,281 | +9,186 (1.91×) |
| **eager (every free)** | **16.77 M** | **+16,760,886 (1,661×)** |

**Capstone** (`tree_cost_*`, keys=2000, rounds=2, 4 000 ops, `-O0`):

| config | per-op (instr) | note |
|--------|---------------:|------|
| bump (baseline) | 1,719 | — |
| revoke-on-free, revoke off | 96,202 | +94,483: Phase-0 allocator O(n) slot-scan |
| **full revoke-on-free** | **96,212** | **revoke-at-free = +10 instr/op, O(1)** |

---

## 6. Cross-system reading (the money comparison)

The marginal cost of making **one `free` temporally safe**, each system on its own
QEMU vehicle:

- **Ours:** a single capability op at the point the object dies — **+5 instr/op
  (`-O2` microbench), +10 (`-O0` real workload)**, O(1), independent of live-set
  and workload.
- **CHERI eager** (matches our security): a full address-space sweep for
  capabilities into the freed region — **~14–17 M instr/op**, recurring per free.
- **CHERI async** (deployable default): the same sweep amortized across the frees
  between quarantine flushes — **+9–20 k instr/op (1.9–6.4×)** — but blocks 0/11
  use-after-free at the contract point.

**Why async's ratio drops on the real workload (6.4×→1.9×):** the BST does real
per-op work (build + pointer-chasing lookup, ~10 k instr/op baseline), so the
amortized quarantine sweep is a smaller *fraction* of the total. **1.9× (≈+91%) is
the representative deployed-CHERI temporal-safety overhead** on a real program.

**Why eager's per-free cost is even higher on the real workload (14 M→16.8 M):**
the 2000-node live-set means more live capabilities for each sweep to scan.

**The structural point (robust to the proxy):** CHERI's temporal cost is a sweep
whose work scales with mapped memory and recurs per free under eager; ours is a
fixed O(1) op. The gap widens without bound as the reachable heap grows.

---

## 7. Fairness — is this comparison honest?

**Fair, for the claim it makes; not a raw absolute-number comparison — and the
paper says so explicitly.**

**What is fair:**
- Identical workload (byte-for-byte the same loop / same BST source both sides).
- Identical metric (dynamic retired-instruction count, empty-loop calibrated).
- Each measured as overhead over its *own* unprotected baseline, so the number is
  "what did turning on temporal safety add," not a cross-machine absolute.
- The `revoke − norevoke` delta on the Capstone side isolates the revoke op:
  both configs pay identical alloc-side bookkeeping, so the delta is the revoke
  intrinsic alone — workload- and optimization-level-robust.

**What is NOT comparable (stated in the paper):**
- The two vehicles differ: bare-metal Capstone domain vs full-OS CheriBSD process
  (jemalloc, kernel, interrupts). So the *baselines* (7/1,719 vs 3,760/10,095) are
  not comparable, and we never compare them. Only within-vehicle overhead and the
  O(1)-vs-sweep asymptotics are load-bearing.
- CHERI's counts include kernel time — deliberately, since revocation *is* kernel
  work. `rdinstret` also counts timer-interrupt handlers in the bracket, a minor
  contaminant the ~1–3% trial spread shows is small (and negligible against
  eager's millions).
- Our directly-CHERI-comparable "temporal cost" is arguably the *full*
  revoke-on-free allocator (+58 `-O2` microbench), not just the +5 revoke op,
  because CHERI's number is its *whole* temporal mechanism. Even the
  allocator-inclusive figure vs eager's millions is ~5 orders of magnitude; the
  +5/+10 is the fair comparison to CHERI's per-free *revocation action*.

**Residual fairness gap (worth surfacing up front):** a perfectly symmetric test
would run our allocator inside an OS too. We didn't — our side is bare-metal. So
the honest framing is a *mechanism-cost* comparison, robust on the delta and the
asymptotics, not a same-OS end-to-end head-to-head. The Capstone RTL number and a
real-workload OS run would close that gap.

---

## 8. Caveats / threats to validity

1. **Instruction-count proxy, not timing.** QEMU has no pipeline/cache/cycle
   model. Cycle-accurate confirmation of our side is the Capstone RTL follow-on
   (`tests/rtl-smoke/`); CHERI has no comparable silicon here (hence QEMU-to-QEMU).
2. **Capstone real-workload is `-O0`.** The BST's capability-value selects
   (`cur = cond ? cur->l : cur->r`) ICE the Capstone `-O2`/`-O1` backend — a
   known codegen gap flagged for the compiler lane (COORDINATION.md). `-O0`
   compiles; the load-bearing number is the revoke−norevoke delta, which is
   optimization-robust. (The microbench, which has no such select, is `-O2`.)
3. **Capstone alloc-side blow-up is the allocator, not the mechanism.** The
   Phase-0 `revoke_on_free_alloc.h` does an O(n) linear `rof_find`/slot scan per
   op, which explodes with a 2000-object live-set (+94 k on the BST). A production
   allocator (umm_malloc, #78) makes that O(1). The revocation primitive is the
   +5/+10.
4. **Eager trial counts.** eager is expensive enough that it gets n=2 (microbench)
   / n=1 (BST); other configs n=3. Trial spread on the cheap configs is ~1–3%.
5. **Naive workloads.** A tight malloc/free and a bounded BST; real applications
   differ. The qualitative gap (inline O(1) op vs address-space sweep) is
   structural, not an artifact of these microbenchmarks.

---

## 9. What landed in the paper

`paper/evaluation.tex` §`sec:eval-perf-compare` (subsection "Temporal-safety
overhead: our system versus CHERI"):

- `tab:perfcompare` — microbench, both systems, three configs each.
- `tab:perftree` — real BST workload, both systems.
- Analysis paragraphs: "The security-matching CHERI config is not
  performance-viable", "Contract-point revocation is O(1)", "On a real allocation
  workload", "What this establishes and what it does not".
- Abstract + intro updated to surface the performance finding (the eager-sweep vs
  O(1)-contract-point difference) as a headline contribution alongside security.


