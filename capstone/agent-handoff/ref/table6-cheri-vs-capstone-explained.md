# Table VI explained — CHERI-RISC-V purecap vs. our system (Capstone)

*Reference companion to `paper/evaluation.tex` (`tab:cheri`). Explains every column,
every cell symbol, and every configuration term, with a minimal C example for each
defect archetype in the corpus, and states plainly where Capstone beats CHERI and
where the two are equal.*

---

## 1. What the table is asking

The table takes the 15-defect corpus of Table III (real lifetime-contract bugs from
CPython, PHP, Ruby, Rust, Go, and JS SQLite bindings), compiles each one for
**CHERI-RISC-V purecap**, and records — per defect — **whether and *when* CHERI blocks
the offending memory access**, placed next to what **our system (Capstone)** does.

The whole point is *temporal* memory safety at a **contract point**: the instant the
API's lifetime rule is violated (a statement is finalized, a connection is closed, a
buffer is reused), not the instant `free()` happens to run. That distinction is the
entire story of the table.

---

## 2. The two models being compared

### CHERI: spatial by construction, temporal by *revocation*

A CHERI capability is a fat pointer (base, bounds, permissions) protected by a
hardware **tag** bit. Spatial safety (out-of-bounds, forged pointers) is enforced *by
construction* — every load/store checks the cap. But a capability to a **freed** object
is still perfectly valid: correct bounds, tag still set. Nothing about the object being
dead is visible to the hardware. CHERI recovers temporal safety by **revocation**:
after `free()`, a sweep invalidates every outstanding capability that points into the
freed region. How aggressively that sweep runs is a *deployment policy*, so the table
measures three:

| Config | What it is | Cost | Deployed? |
|--------|-----------|------|-----------|
| **spatial** | bounds + tags only, **revocation OFF** | free | yes, but no temporal safety |
| **async**†  | revocation ON, **quarantine-driven lazy sweep** (CHERIvoke / Cornucopia) | amortized | **yes — the realistic default** |
| **eager**   | **revoke on every `free()`**, synchronous | very expensive | no — an upper bound |

† `async` is the realistic deployed default because a synchronous sweep on *every*
`free()` (eager) is prohibitively slow — CHERIvoke/Cornucopia exist precisely to
*defer* and *amortize* revocation via a quarantine. So the honest question is: **what
does the *deployable* config (async) catch?** Answer: for the temporal class, nothing
at the contract point.

### Capstone: the contract point *is* the revocation point

Capstone provides linear-capability primitives that let the lender revoke a borrowed
capability **at the logical lifetime boundary itself**, synchronously, in O(1), with no
sweep. The primitive letters in the last column (from Table `tab:fix`):

| Letter | Primitive | Enforces |
|--------|-----------|----------|
| **L** | **Linear** capability | unique ownership; a lent handle cannot be duplicated/retained |
| **R** | **Revocation** capability | invalidate a delegated capability at the contract point |
| **H** | **Hierarchical** capability | child cap dies when the parent (db/connection) is destroyed |
| **U** | **Uninitialised** capability | memory cannot be read before it is validly initialised |
| **S** | **Sealed** capability | opaque handle a callee cannot forge/dereference out of contract |

Because Capstone revokes **at the contract point** and not at `free()`, it (a) faults
*synchronously* when the bug happens and (b) still catches bugs where **nothing is ever
freed** — which no CHERI policy can.

---

## 3. The cell notation

| Symbol | Meaning |
|--------|---------|
| `--` | **not blocked** — the defect survives and runs to completion (silent corruption) |
| `late` | **faults, but only *after* the `free()`** — caught at the dereference because eager revocation invalidated the cap at free time; never keyed to the logical contract |
| `abort` | caught by the **allocator's own double-free abort**, *not* by revocation |
| `sync` | **faults synchronously at the offending access** — the good case |
| `L,R,H,U,S` | Capstone: faults synchronously; letters name the primitive(s) used |

---

## 4. The defect archetypes, with C examples

The 15 corpus rows collapse into a handful of archetypes. For each: a minimal C
reproducer, CHERI's three verdicts, Capstone's verdict, and who wins.

### 4.1 Use-after-free — the core temporal case (rows 1, 2, 3-as-shimmed, 6, 15)

```c
sqlite3 *db;              sqlite3_open(":memory:", &db);
sqlite3_stmt *stmt;
sqlite3_prepare_v2(db, "SELECT 1", -1, &stmt, NULL);

sqlite3_finalize(stmt);  // CONTRACT POINT: the statement object is freed here
sqlite3_step(stmt);      // BUG: dereferences the freed statement
```

| Config | Verdict | Why |
|--------|---------|-----|
| CHERI spatial | `--` | the cap to `stmt` still has valid bounds + tag; freedom is invisible → stale read succeeds |
| CHERI async (**default**) | `--` | the quarantine sweep has not run; the dangling cap is still tagged at `step()` → **the bug happens uncaught** |
| CHERI eager | `late` | revoke-on-every-free killed the cap at `finalize()`, so `step()` traps — but only because of the non-deployable full-sweep-per-free policy |
| **Capstone** | **`sync` (L,R,S)** | the borrowed handle is **revoked at `finalize()`** (the contract point); `step()` faults deterministically, O(1), no sweep |

**Winner: Capstone.** The *deployable* CHERI config (async) misses this entirely. Only
the prohibitively expensive eager config catches it, and even then merely as a
side-effect of the physical free.

### 4.2 Use-after-close / destruction-order (rows 4, 5, 7, 8, 9, 10)

```c
sqlite3 *db;             sqlite3_open(":memory:", &db);
sqlite3_stmt *stmt;
sqlite3_prepare_v2(db, "SELECT 1", -1, &stmt, NULL);

sqlite3_close(db);       // CONTRACT POINT: connection destroyed; stmt is now orphaned
sqlite3_step(stmt);      // BUG: child handle used after its parent died
```

Same CHERI verdict pattern as 4.1 (`-- / -- / late`). Capstone uses the **hierarchical**
primitive **H**: the statement capability is a *child* of the connection capability, so
destroying the connection revokes the child in the same operation. `step()` then faults
synchronously.

**Winner: Capstone**, and note the *structural* benefit: H expresses "child dies with
parent" as a first-class relation. CHERI has no notion of the parent/child lifetime
link — it can only observe the raw `free()` of whatever `close()` happens to release.

### 4.3 Double-free (row 11)

```c
sqlite3_stmt *stmt;      /* ... */
sqlite3_finalize(stmt);
sqlite3_finalize(stmt);  // BUG: frees the same object twice
```

| Config | Verdict | Why |
|--------|---------|-----|
| CHERI spatial | `--` | double-free is a temporal error; bounds/tag say nothing |
| CHERI async / eager | `abort` | caught, but by the **allocator's double-free detector**, *not* by CHERI revocation — the same abort a normal libc gives |
| **Capstone** | **`sync` (L)** | the **linear** capability was *consumed* by the first `finalize()`; a second use has no live capability to present → faults |

**Winner: even here Capstone is cleaner** — the `abort` is not a CHERI property; it is
the allocator. Capstone's linearity makes the second free a *type-level* impossibility
(the handle was moved out), not a runtime heuristic.

### 4.4 Null-dereference and uninitialised (rows 12, 13, 14) — *CHERI already wins these*

```c
sqlite3 *db = NULL;      // never opened, or row_factory deleted, or Connection uninit'd
sqlite3_step((sqlite3_stmt*)db);   // BUG: deref of null / uninitialised handle
```

| Config | Verdict | Why |
|--------|---------|-----|
| CHERI spatial / async / eager | `sync` in **all three** | a null / uninitialised slot has **no valid tag**; the very first deref traps by construction — no revocation needed |
| **Capstone** | `sync` (L,R for row 13; H for 12; **U** for 14) | same outcome; row 14 specifically uses the **uninitialised** primitive so reading-before-init faults |

**Tie.** *This is the intellectually honest part of the table:* for the spatial / null /
uninitialised rows, base CHERI is already sufficient — both systems catch them
synchronously. Capstone claims **no** advantage here. Its advantage is confined to the
*temporal* class (4.1–4.3, 4.5), which is exactly where the deployable CHERI config
fails.

### 4.5 Reuse-not-free — the row *no* CHERI config catches (row 3r)

This is the **real** upstream diesel bug (row 3's true shape; row 3 in the corpus is a
*shimmed* UAF, 3r is the genuine defect).

```c
sqlite3_stmt *stmt;      /* ... */
sqlite3_step(stmt);
const unsigned char *p = sqlite3_column_text(stmt, 0);  // borrow engine-owned buffer

sqlite3_step(stmt);      // CONTRACT POINT: the engine REUSES the same buffer IN PLACE
                         // -- nothing is freed; the object is alive the whole time
printf("%s", p);         // BUG: p now reads the NEXT row's bytes (stale read)
```

| Config | Verdict | Why |
|--------|---------|-----|
| CHERI spatial | `--` | in-bounds, valid tag |
| CHERI async | `--` | no `free()` ever happens → revocation is **never even triggered** |
| CHERI eager | `--` | same — revoke-on-every-free is irrelevant when there is no free |
| **Capstone** | **`sync` (R)** | the borrow is **revoked at the `step()` that reuses the buffer** — the contract point — regardless of free |

**Winner: Capstone, decisively.** This is the clean **"CHERI cannot."** CHERI's entire
temporal-safety mechanism is keyed on `free()`; a stale-but-allocated read where the
object is *never freed* is outside its model by design, at *any* policy. Capstone
catches it because its enforcement is tied to the **logical contract** (the buffer is
lent only until `step` advances), not to the allocator.

---

## 5. The two tally rows — what the table proves

```
                                               spatial  async†  eager   Capstone
  Blocked at all (of 15)                          3       4       15       15
  use-after-free/-close, at the contract point    0/11    0/11    0/11     11/11
```

- **Blocked at all:** spatial catches only the 3 null/uninit rows (12/13/14). async adds
  1 (the double-free *abort*, not revocation) → 4. eager reaches 15 because
  revoke-on-every-free eventually invalidates every dangling cap — but every temporal
  catch is `late`, and it is **not a deployable policy**. Capstone: 15, all `sync`.

- **The money row — UAF/use-after-close at the contract point:** there are 11 such rows
  (1–10 + 15). **Every CHERI configuration catches 0 of them at the contract point.**
  Even eager only faults `late`, after the free. Capstone catches **11/11**
  synchronously. This single line is the security argument.

- **And row 3r sits *below* the 15:** the real reuse-not-free defect that **no CHERI
  policy catches at all** (`-- / -- / --`), while Capstone catches it with R.

---

## 6. Benefits of Capstone over CHERI (summary)

> **Framing update (PI, 2026-07-14): security is a near-tie at the eager config;
> the real axis is performance.** The `eager` column shows CHERI *can* match our
> security (it blocks all 15). It does so only via a stop-the-world revocation
> sweep on every free (Cornucopia — "a very slow version of what Capstone does,
> like a garbage collector"), which is why it is not deployed. We realize the same
> revoke-at-free as an **O(1)** op. So on the injected corpus the security
> comparison is a **near-tie at the non-deployable eager config** (row 3r aside);
> the distinguishing claim is that **we achieve it far more cheaply** — quantified
> in the separate QEMU-to-QEMU overhead table (`plans/perf-cheri-vs-capstone-qemu.md`,
> paper `sec:eval-perf-compare`). Also: the row-11 `abort` is **discounted**
> (software detection, no capability check). The points below still hold, but read
> them through this lens — the temporal-*coverage* wins (async gap, 3r) plus the
> *cost* win, not a blanket "more secure than CHERI."


1. **Catches the deployable-config gap.** The realistic CHERI default (async revocation)
   blocks **0/11** temporal defects at the contract point. Capstone blocks 11/11. To
   even approach Capstone's coverage, CHERI must run **eager** revoke-on-every-free — a
   policy nobody deploys because of its cost.

2. **Synchronous, at the logical contract point** — not deferred to a stop-the-world
   sweep (async) and not bound to the physical `free()` (eager). The fault fires *where
   the program logic actually violates the lifetime rule*, which is also where a
   developer needs the diagnostic.

3. **Catches reuse-not-free (3r), which CHERI cannot at any policy.** When a live,
   never-freed buffer is reused in place, CHERI's free()-keyed revocation is simply not
   in the loop. Capstone's contract-point revocation is.

4. **O(1), no sweep, no quarantine.** Revocation is a single capability operation, not a
   memory-scanning GC-like pass. No amortization tricks, no quarantine memory blowup, no
   pause.

5. **Lifetime relationships are first-class** (H = child-dies-with-parent, L = linear
   move/consume, S = sealed opaque handles). These express the *binding contract*
   directly, so bugs like destruction-order and double-free become type-level
   impossibilities rather than runtime heuristics.

**Where Capstone claims nothing extra (honest scope):** the null-deref and
uninitialised rows (12/13/14). Base CHERI already faults on those by tag construction;
both systems are equal there. Capstone's win is precisely and only the **temporal
class**, which is the class the corpus is built from and the class the deployable CHERI
config misses.

---

## 7. One-line intuition

> **CHERI asks "does this pointer have valid bounds and a live tag?"**
> **Capstone asks "is this borrow still within its contract?"** — and revokes the moment
> it isn't, whether or not anyone called `free()`.

---

*Sources: `paper/evaluation.tex` (`tab:cheri`, `sec:eval`); CHERI baseline data
`capstone/tests/cheri-baseline/RESULTS.md`; primitive definitions Table `tab:fix`.*
