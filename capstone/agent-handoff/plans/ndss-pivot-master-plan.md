# Master plan — NDSS pivot (from the 13 July design meeting)

**Status:** active, drives all work from 2026-07-13. Supersedes the "characterization+design,
defer eval" framing. Sync cadence: short calls; next full sync ~Friday. Coordinate on the
shared Slack channel.

**Deadline:** target **NDSS**, ~6 weeks out. PI wants **~90% of the paper by end of July**
(PI travels Aug 4 + teaching after). So the next two weeks are the push; PI helps heavily on
writing.

---

## 1. What the paper actually is now (the reframe)

Not "pointer-safe SQLite marshalling (design/position)." It is a **systems-security paper on
a new hardware-assisted abstraction for safe cross-domain pointer sharing** between two
mutually-distrusting runtimes — *"like HFI, but better,"* the extra being **hardware
cross-boundary pointers**.

- **Threat model.** You compile your code with our compiler; the binary links against an
  **untrusted** binary (e.g. SQLite) that a third party compiled with the same compiler.
  **Import/export tables are fixed in the binary interface**: which functions each side may
  call, which memory regions each may access. We uphold the control-flow + memory-access
  restrictions. The untrusted side just links and runs.
- **Mechanisms (the interface primitives, = the L/R/H/U/S table).**
  - **L linear** — carries cross-boundary pointers; security rests on linear caps (alias-free
    handoff of a shared region).
  - **R revoke / H hierarchical revoke** — end the borrow / cascade on teardown.
  - **U uninitialised** — safe reclaim / use-before-init.
  - **S sealed** — control-flow enforcement for **callbacks**: the PC is stored in a sealed
    capability; the other side can only call back through it.
- **Two technical parts.** (i) **Hardware complexity** of implementing these checks — the RTL
  design **already exists** (built by the hardware collaborator); we borrow its implementation
  chapter. (ii) **Software** = our compiler pass; position it against what CHERI already does.
- **Case study = SQLite.** Embedded in many hosts, historically full of cross-boundary pointer
  bugs; our system removes them; we measure the overhead.

**Correction to prior belief:** there IS real RTL, and "works on QEMU ⇒ works on RTL" is
expected to be a small gap. So the perf vehicle is **not** blocked — RTL/FPGA is available now
(access via the hardware-access contact's web deploy interface). B's task-014 instruction-count
proxy is the *interim* number; real hardware numbers are the target.

## 2. Two storylines

**A. Security** — the CVE class. For each corpus row: show it is **exploitable / not blocked
under the baseline** (HFI, and CHERI), and **stopped on our system**. Doable on QEMU.

**B. Performance** — end-to-end cost on RTL/FPGA of: micro-benchmarks (**BEEBS, RV8, CoreMark**;
maybe IOzone) + the **SQLite CVE benchmark** + **SQLite's own built-in tests**. Report
**aggregate first**, then a **breakdown** (spatial / temporal / linear) by *selectively disabling
features* (e.g. make it non-linear, skip checks, re-run). PI's prior: the **linear-capability /
borrow tree is the main overhead** — that component is the one to isolate.

## 3. Baselines (this is the crux)

- **CHERI = #1 competition.** Reviewers will demand it. Story = *linear capabilities vs CHERI's
  revocation-sweep (Cornucopia / MTE-style) temporal safety*. CHERI's known gap: after
  free+realloc you can still reach the object, and its temporal defense is a **stop-the-world
  sweep** that validates all downward pointers — expensive, and it does **not** fire on the
  **lifecycle contract points** (`step`/`reset`/`finalize`/`close`) our defects violate when the
  memory is not actually freed. CHERI also has no first-class **hierarchical senior-revoke** and
  no **sealed cross-domain callback** as an isolation primitive.
- **HFI = secondary, mostly conceptual.** HFI isolates *domains* (spatial + enter/exit); it is
  **not expressive enough to distinguish alive vs. dead**, and the buggy glue is *trusted code
  inside its own domain*, so isolation cannot help. Show conceptually it can't capture the class;
  optionally run **GEMFI** (x86 QEMU HFI) on a couple of repros as evidence it doesn't stop them.
  (No RISC-V HFI code exists — the RISC-V "standard" repo is docs only.)

## 4. Corpus trim

Remove the out-of-scope rows so the count is defensible (reviewers stop complaining ~15–20).
Meeting explicitly named **15, 18, 19** out, "PHP [stale-state] goes out, CPython covered by
other examples," target ≈ **15 rows**. That lands on removing **rows 15, 17, 18, 19** (my
out-of-scope tier: subinterpreter deadlock, node typecast abort, PHP stale-state ×2) → **19 → 15
rows**. Keeps PHP UAC/UAF rows 4,5,6. *(Row 17 inferred to hit the 15 target; confirm at sync.)*
Rewire `tab:scope` / `tab:fix` / `tab:valid` + the prose that leans on removed rows (esp. the
row-15 deadlock "no memory model addresses" talking point).

## 5. Naming

A name suggesting HFI-but-better; **"cross-domain" / "cross-border" is the keyword.** A proposes
candidates (not urgent). Seeds: *XDomain / CrossGuard / BorderCaps / Interlace / Passport /
Threshold*.

## 6. Work breakdown (owners, sequence)

**Lane A = me** (paper, security narrative, corpus, repros, per-row oracle).
**Lane B** (`capstone-bootstrap-b`) = compiler/emulator/measurement/**hardware**.

### This week (critical path — "get the security section written")
| ID | Work | Owner |
|----|------|-------|
| **T1** | **CHERI corpus test (Task 1):** build corpus under CHERI (purecap) on CHERI-QEMU (cheribuild), run each repro, record **blocked / not-blocked + why**. Expect CHERI misses the temporal/stale-handle + sealed-callback cases. | **B** (A supplies the per-row oracle) |
| T2 | **RTL/FPGA smoke** while the hardware collaborator is reachable: compile an existing repro/probe on the RTL, run, confirm + first perf number. Time-sensitive. | **B** (+ hw-access contact) |
| T3 | **Security section draft** — CVE class; HFI-can't (conceptual); CHERI-can't (fill from T1); our-system-stops. Scaffold now, fill CHERI cells post-T1. | **A** |
| T4 | **Remove out-of-scope rows** (15,17,18,19 → 15) across tables + prose. | **A** |
| T5 | **HFI conceptual argument** (+ optional GEMFI run for evidence). | A (+ B for GEMFI) |

### Weeks 1–2
| ID | Work | Owner |
|----|------|-------|
| T6 | **Embedding interface + host shims:** define import/export interface; per CVE, replace the big host app with a **minimal vulnerable shim** exercising the boundary. This is the real system under test. | **B** (A defines per-CVE boundary contract) |
| T7 | **Performance eval on RTL/FPGA:** aggregate end-to-end (micro + SQLite CVE bench + SQLite built-in tests). | **B** |
| T8 | **Related-work section:** CHERI (#1), HFI/SFI, Cornucopia/MTE, Keystone, CapsLock, MPK. | **A** |
| T9 | **Paper restructure:** from design-paper to full paper (threat model, abstraction, hardware-complexity, compiler, security eval, perf eval). | **A** (PI co-writes) |

### Weeks 3–4
| ID | Work | Owner |
|----|------|-------|
| T10 | **Overhead breakdown** (spatial/temporal/linear via feature-disable); confirm borrow-tree dominates. | **B** |
| T11 | Name lock-in; full draft to 90%. | A + PI |

## 7. Open items for the Friday sync
1. Exact removed rows — confirm **15,17,18,19 → 15 rows** (17 inferred).
2. NDSS cycle / the ~6-week deadline date.
3. CHERI comparison config (purecap vs. benchmark ABI; both modes?).
4. Whether the RTL smoke (T2) happens this week (hardware collaborator's availability + access).
