# Current recommended next step

## 2026-08-01 (early) — READ THIS FIRST: results on this board are NOT always reproducible

### The finding that changes how everything else must be read

Determinism had never been tested in this campaign. It has now. Running the SAME binaries
repeatedly inside ONE boot:

    wd66  x3   rc = 2, 2, 2          DETERMINISTIC
    wd63  x2   rc = 0x0E, 0x0F       NON-DETERMINISTIC
    fn63  x2   rc = 0x0F, 0x0E       NON-DETERMINISTIC (confirms it)

**`wd63` returns different answers on identical back-to-back runs.** Consequences:

1. **"The first walk succeeds, later ones fail" is RETRACTED** — it was built on `wd63 = 0x0E`,
   and the next run gave `0x0F` (array 0 overran too).
2. **Every single-sample conclusion in this campaign is unsafe**, including
   `stage 52 = 0xC1` ("`lit[1]` is the bad one"), which drove days of bisection. A result seen
   once is a sample, not a fact.
3. **Run every probe at least 3x from now on.** The runner accepts the same `.dom` repeated in
   `SQLITE_STAGE_DOMS`; repetitions inside one boot are nearly free.

### What is genuinely established (and reproducible)

* **It is a LIVELOCK, not a hang.** Stage 51 returns `0xB1` — the domain runs and RETURNS.
  Every hypothesis requiring the core to stop is retired.
* **The emitted pointers are provably correct.** `__capstone_cap_init` derives the literals at
  `0x6da / 0x6e0 / 0x6e6` — deltas of exactly 6 — matching the merged `.rodata` container.
  1544 straight-line instructions, zero calls, zero branches; the one reused register is
  correctly spilled and reloaded. (Proof about EMITTED code, not about runtime values.)
* **`wd66` is a deterministic reproducer** (5 samples, all `2`): the same element walked TWICE
  through the SAME pointer — first walk overruns, second terminates correctly. Its two walk
  loops were verified byte-identical (23 instructions each, `0x36994` / `0x36a40`, only branch
  targets differ). Use `wd66` as the vehicle for any further bisection; it is the only stable
  failing case found.

### Refuted BY MEASUREMENT (do not revive without new evidence)

| hypothesis | how it died |
|---|---|
| `cincoffset` consumes its source | `flu_unit.anvil:43,:62` return `rs1` unchanged |
| `STC` clears its source register | `dyn_unit.anvil:427` returns `rs2_v` unchanged |
| carve/rev-node pool exhaustion | 183 carves measured against a ~1000 budget |
| `LDC` consumes its memory slot | stage 57/58 = 7 (two reads, both non-NULL and equal) |
| the SHA5 wedge is self-inflicted | UNGUARDED `wd51` returned `0xB1`, unchanged |
| array identity ("Nth array is broken") | `wd60/61/62`, one shared array, only the loop failed |
| granule misalignment is the root cause | `ga60 = 0xC1`, identical with granule-aligned glue |
| "first walk succeeds" | `wd66 = 2` inverts it; `wd63` varies anyway |
| store ordering / missing fence | `fence rw,rw` before `domain_main`: `fn66 = 2`, no change |

### Real but LATENT (fix on its own merits, NOT this bug)

* **Carve base granule misalignment.** idx 170, `sqlite_heap`, 262144 B, granule 512,
  `base % g = 64`, `len % g = 0`. Simulation over the real descriptor: granule-align OFF -> 1
  unrepresentable carve, ON -> 0, for every plausible region top. **The 2026-07-31 revert note
  had the failing end backwards** (it blamed the length; the length is fine). `ga60` shows
  enabling it does not fix the livelock, so it is a separate correctness issue.
  Knob: `INTERP_GRANULE_ALIGN=1`.
* **`wd65` wedges where `wd62` returns 5** — same array, same single walk, differing only in a
  `volatile` pointer load, and the failure modes differ (domain death vs overrun-and-return).
  Open thread; do NOT assume it shares a cause with the livelock.
* Domains run with `mtvec = ctvec = 0` (no monitor writes `dom_seal[1]`) — upstream design
  question, deliberately not patched unilaterally.

### Next step

Re-take the foundational bisection results WITH REPETITION, starting with stage 52, and treat
any result that varies as unusable until characterised. `wd66` is the stable vehicle for
narrowing the livelock itself.

### Tooling and traps (all of these bit during 31-07/01-08)

1. **Never read `board-<tag>.log` for results** — it carries the accumulated console
   scrollback, so it returns markers from EARLIER runs. Only `PROBE_SCOPED_OUT` is valid.
2. **Never pattern-match a string your own command line contains.** `while pgrep -f "make
   build LINUX_PAYLOAD"` matched itself and deadlocked six shells for ~50 minutes while
   reporting false progress. Use a bracket pattern (`"[m]ake ..."`).
3. **A domain earns an early slot only if THAT EXACT BINARY has returned before.** Guarded
   `wd53` and `wd65` were placed early as "controls" on source-level identity after the binary
   changed underneath; each wedge ended its run.
4. **`llvm-objdump --disassemble-symbols` silently truncates** (~470 of 9088 bytes, stopping at
   a local `.Lpcrel` label). Use `--start-address/--stop-address` and check the disassembled
   size against the symbol size.
5. **Prune only your OWN staged domains** — never package-installed ones (`fib`, `sbi`,
   `smode`, `thread`), which desyncs buildroot's stamps (six boot failures). Keep
   `sqlite_silicon.dom` and `sqlite_host.user` for the freshness gate.
6. **Each staged block's statics land in EVERY build** unless `#if`-guarded — `stage` is a
   function parameter and probes build at `-O0`, so nothing folds. Guards are in place for
   stages 51-66; keep adding them.
7. Build probe batches with `build-stage-probes.sh` — it prints per-artifact hashes and a
   distinct-hash count, so a silently-cached build cannot pass as fresh.
8. Image size: 10.5-15.4 MB boot fine; 26 MB and 46 MB do not.

## STRUCTURAL LIMIT ON SAMPLING (learned 2026-08-01, affects how to read every result)

**A wedge ends the board session, so a WEDGING domain can never be repeated inside one boot.**
Every "n=2/n=3, deterministic" figure recorded above is therefore necessarily from a
RETURNING domain (`wd66`, `wd61`, `wd62`, `wd63`). Every wedging result — `wd10`, `wd52`,
`wd53`, `wd65`, `wd67` — is a SINGLE sample by construction, not by choice. "Wedges
consistently" has never actually been established for any of them.

Consequences for method:

1. To repeat a wedging case, use **separate boots** (~5 min each). Batch the returning probes
   within a boot; batch the wedging ones across boots.
2. **Prefer a probe that RETURNS a marker over one that wedges** — that is what the stage-51
   watchdog achieved (silence -> `0xB1`) and it is what made any of this measurable. When
   bisecting a wedging stage, build the bounded/early-return variant FIRST.
3. Do not describe a wedge as reproducible without naming how many boots it was seen in.

Also: **stage N ⊃ stage M for M < N on the normal path.** Stage 3 (`sqlite3_open`) contains
`sqlite3_initialize`, which contains `sqlite3RegisterBuiltinFunctions` (stage 10). Ordering
stage 3 before stage 10 guarantees the run dies before reaching 10. Order staged probes so a
superset never precedes the subset it depends on.

Also: **never wait on a process by name.** Three separate deadlocks were caused by a command
polling `pgrep -f <pattern>` where its own command line contained the pattern — including once
where a bracket pattern (`"[b]uild-..."`) still matched because the same script later invoked
the real string. Six shells hung ~50 minutes on the first occurrence while reporting false
progress. Sequence steps inside ONE script instead of polling for another task.

## The blocker is SOLID, not intermittent (3 separate boots, 2026-08-01)

Because a wedge ends its session, stage 10 was sampled across three SEPARATE boots, each
running `wd66` first as a liveness control:

    boot1: WEDGED    boot2: WEDGED    boot3: WEDGED
    samples=3  successes=0  ->  ALWAYS FAILS

**`sqlite3RegisterBuiltinFunctions` fails every time.** This closes the possibility raised by
`wd63`'s run-to-run variation that SQLite might sometimes get through — there is no retry path
to an existence proof. It is also the first wedge in this campaign established as reproducible
rather than assumed from a single sample.

Note what this does and does not say: the BLOCKER is deterministic across boots, while some
PROBES (`wd63`) vary within a boot. Both are true; they are different quantities. The
non-determinism does not rescue the blocker, and it does not excuse the earlier single-sample
conclusions either.

### Where to resume

`wd66` remains the only stable failing reproducer (7 samples, all `2`): the same element walked
twice through the same pointer, first walk overruns, second terminates, with the two loops
verified byte-identical. Narrowing that is the live thread — it is small, deterministic, and
sits in the same code family (data-dependent string walk) as the blocker.

Do NOT resume by re-deriving "lit[1] is broken": that came from `stage 52 = 0xC1`, a single
sample that could not be re-taken (the guarded rebuild wedges), and `wd62`/`wd59` both show
`lit[1]` walking correctly in isolation.
