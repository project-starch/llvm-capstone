# S-12 — what is DEAD, and what each refutation does NOT cover

**Why this file exists.** `00-README.md` is 2,200 lines of narrative in the order things were
learned. That is the right shape for evidence and the wrong shape for the question people
actually arrive with — *"has this already been tried?"* On 2026-08-27 the compiler lane and the
RTL lane independently re-derived the same already-recorded chain **on the same afternoon**, and
three separate prior-art hits landed in one session. This is the index.

**Read the fourth column.** A refutation is a fact about ONE experiment, not a permanent
exclusion. The project rule is *read PAST the root cause* — a fixed issue's folder records what
its fix did not cover, and reading only the headline gave the wrong answer twice. So every row
says what it leaves open.

**Rule for adding a row:** name the evidence (a commit, a test, a `file:line`), not a conclusion.
A row with no evidence pointer is an opinion and will be treated as one.

---

## Dead: image layout and geometry

| Hypothesis | How it was refuted | What that does NOT cover |
|---|---|---|
| The **function's address** decides it | `TEXT_PAD` build placed at the "curing" address still wedges. Retracted **three times** under "layout decides it" | Address is not the variable. It does NOT say layout is irrelevant to the *rate* — per-image clustering is real and unexplained |
| The **globals region** decides it | `SQLITE_GOFF_OVERRIDE` moved it +64 KiB; still wedges | Only the region's *base*. Cap-table geometry is a separate knob |
| The **cap-table entry count** decides it | 338 entries wedges, 337 completes — no monotone relation | Counts. A specific *index* colliding is untested |
| The **stack budget** decides it | 16 → 24 moved only the stack; no flip in outcome | Stack size. Slot *contents* and frame overlap were addressed separately |
| The **slot address** decides it | `gp6` moved the slot; still wedges | The slot's address. Not its alignment or granule sharing |
| The **`.bss` size** decides it ("32 bytes cures it") | **Superseded**, not refuted: the apparent cure was draw variance. The real figure is a **54% per-draw rate** with per-image clustering | Nothing was measured wrong; the *inference* over-read N. Any future "X cures it" needs n large enough for 54% |

## Dead: the instruction window

| Hypothesis | How it was refuted | What that does NOT cover |
|---|---|---|
| The **four-instruction shape** is sufficient | Arm 7 reproduces production's exact shape at production spacing and **returns clean** | The shape in isolation. It does not exclude the shape *plus* an untested precondition |
| **Cache residency** (a cold load) is the missing ingredient | Arm 8 = arm 7 with the line evicted first. Returns `0xC12A8000` clean, 245,623 cycles vs arm 7's 1,924 — the walk provably ran, so the load genuinely missed | Miss latency as *modelled here*. Real misses under memory pressure, interrupts and rev-node churn are not the same thing |
| **Instruction spacing** is the variable | Arm 3 added one nop before the consumer; no flip | Spacing at that granularity |

**Standing conclusion for this family:** shape, spacing and miss latency **together** are
insufficient. The trigger needs something the window does not carry — execution history, the
capability's provenance, i-cache/code-volume effects, or a SQLite path unlike the modelled one.

## Dead: mechanisms in the pipeline

| Hypothesis | How it was refuted | What that does NOT cover |
|---|---|---|
| **Tag loss** — a detagged but otherwise correct capability | `decompress_cap_tagged` (`ariane_pkg.sv:766-782`) passes the **cursor through unchanged** on an untagged read, so tval would be NON-zero. Measured tval is **0** | Tag-only mechanisms **at this site**. Says nothing about sites where tval was not captured |
| **LDC move-clear** fired on the subject | Value is NONLIN, confirmed three independent ways; the clear set is keyed to REVOKE (`load_unit.sv:225-226`) | The subject's type as measured. A type *change* mid-flight is not excluded by a static read |
| **Wrong-producer forwarding** | Prevented at issue: `issue_read_operands.sv:1478` gates `issue_ack` on `!stall_waw`; `:1418` defaults all-ones; `:1427-31` clears only when `rd_clobber_gpr[rd]==NONE` | The generic forwarding path. Capstone-specific result packs are separate |
| **Granule row** — the clear reordered after the next store | **Structural, no boot needed:** the clear shares the store-buffer port (`store_unit.sv:449`), both queues are strict FIFO with monotonic pointers, and `load_unit.sv:707-712` holds `valid_o` low until the clear is accepted. One FIFO, program order. All three premises verified at the resident revision | Reordering *within* the write buffer. Does not cover merge-time `ctag` behaviour, which is S-09/S-10 territory |
| **Register-file row** — a stale FLU operand read before the load lands | `s12-flu-raw.S`: window **proven created** (`flu-issues=131`, `ldc-pending-cycles=82`) and `HAZARDS=0`. The generic RAW machinery stalled it every time | DYN→FLU only, in bare M-mode. An earlier run of the same test was VOID at `ldc-pending-cycles=0` because every load hit — the totals are what make this zero admissible |
| **Register shadow staleness** — an ALU write leaves old capability metadata behind | `alu-write-clears-shadow.S` (`capstone-ariane eb43f5d09`): the capability-then-overwritten register **retires** through `CINCOFFSET`, while the still-a-capability arm **traps** as positive control | Register shadow only. Says nothing about the memory path, and it is bare M-mode |
| **FLU → LSU adjacency** — an LSU consumer reading an FLU producer's dest before it lands | `s12-flu-lsu-raw.S` (`capstone-ariane 21842a864`): no exception, correct cursor `0x80003020`, type query reads NONLIN. Made admissible by evicting first — 2560 `lbu`, exactly 40960/16, 33,677 cycles vs 373 warm | Bare M-mode, single iteration, producer's own operand not pending |

## Retracted claims — asserted, then withdrawn

These were **stated as findings** and are wrong. They are listed separately because a retracted
claim propagates further than a refuted hypothesis.

| Claim | Why it fell |
|---|---|
| "**Two levels wedge**" (deterministic) | It is a **54% rate**. `q_two` never ran in up21–24 — preflight listed it unused in all four — and up24 was a clean completion filed as void. Corpus: 25 wedged / 21 returned |
| "`+0x8c` **is the first executable statement**" | Three initialised declarations precede it |
| "**Layout decides it**" | Retracted three times |
| "The **byte-identical body**" and "**NONLIN measured at the fault site**" | Both retracted 2026-08-19, `00-README.md:1366` |
| "**Memory holds the correct capability**" (as an argument about the load) | The shadow-tag half of that evidence was withdrawn as evidence about the load |
| "The last committed instruction is in `sqlite3_result_double`" | Retracted before publication 2026-08-27. Identical `commit pc` from two images holding **different instructions** at that address ⇒ the commit-pc aperture is stale by construction at a wedge |

## Not S-12 at all

| | |
|---|---|
| The `-O1` hang | **[S-13](../S13-o1-dyn-rev-node-hang/)**. `ex_commit.valid = 0` (no exception), aperture 225 `0xd5` (three wait conditions). S-12 is `ex_commit.valid = 1`, `225 = 0x80`, nothing waiting |
| `caplifive_s10fix_80843404c.bit` as an S-10 control | **The filename is wrong.** `80843404c` is a synth-guard tooling commit that PREDATES every S-10/S-10b commit; `store_buffer.sv` hashes differently from the fix. It carries the S-07 fix and its census is clean, but it cannot test S-10 in either direction. The S-10 RTL fix has never been synthesised |

## Still alive

- **The sequence** that produces the S-13 wait state (RTL lane, generated FSM `7572-7790`).
- **Per-image clustering** — real, unexplained, and the reason redraws must be distinct images.
- **The rate** for the extended workload, which contains a two-table join (`sqlite_capstone_domain.c:1439`)
  and passed 3/3 — where a 54% per-draw rate would predict ≈0.10 for that outcome.
