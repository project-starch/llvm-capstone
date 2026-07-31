# Current recommended next step

## 2026-07-31 (late, second half) — the livelock is localised; codegen is EXONERATED

### Where the SQLite blocker actually stands

The bisection reached `sqlite3RegisterBuiltinFunctions`, and inside it the failure is now a
**LIVELOCK, not a deadlock**. Stage 51 (a bounded `strlen`) returned **`rc = 0xB1`**: the
domain RAN and RETURNED. That retires every hypothesis requiring the core to stop, including
the RTL load-syncer arming leak.

Localisation from there:

* stage 52 = `0xC1` — `lit[1]` is the first literal whose walk never terminates.
* stage 53 = `0xDF` — `lit[0]`'s first 8 bytes are `l t r i m \0 r t`. Correct for a MERGED
  container (an earlier prediction of `0x1F` was wrong: only byte 5 is the NUL; bytes 6,7
  are the next literal). So `lit[0]` is fine AND `lit[1]`'s bytes are demonstrably present.

### What was PROVED offline while the board was down (do not re-litigate)

Full trail: `history/31-07-2026_22-40-00_capinit-literal-leaves-codegen-is-correct.md`.

1. **The emitted pointers are correct.** `__capstone_cap_init` derives the literals with
   `cincoffsetimm` at `0x6da / 0x6e0 / 0x6e6` — deltas of exactly **6 and 6**, matching the
   merged `.rodata` container at `0x16e52e`. The 16 capabilities feed THREE `lit` arrays
   across **1544 instructions with zero calls and zero branches**; the only reused register
   (`a0`) is correctly spilled to `0x260(sp)` and reloaded. **Cap-init is not producing bad
   pointers. Stop looking there.**
2. **`cincoffset` does NOT consume its source** (`capstone_flu_unit.anvil:43,:62` return
   `rs1` unchanged). The theory that `lit[0]` survives and everything after it derives from
   `cnull` fits the symptom perfectly and is nonetheless WRONG.
3. **`STC` does NOT clear its source register** for LINEAR/NONLIN
   (`capstone_dyn_unit.anvil:427`); only the UNINIT path nulls it. The documented linear
   clearing is on **LDC**, and it clears MEMORY, not the register.
4. **Carve exhaustion is not in play.** Measured from `.capstone_gp_initdesc`: 183 carves
   (`wd54/55/56`), 184 (`wd57/58/59`), 179 (`sqlite_silicon.dom`) against a ~1000 budget.
   The 1059 figure belongs to the FULL SQLite build, not to staged probes.

### The live hypothesis (probes built, stages 57-59)

`LDC` clears its MEMORY source when the loaded capability is linear. Stage 52 walks `lit[i]`
by LOADING each element out of the array; stages 53/54 name `lit[0]`/`lit[1]` directly, where
the value can stay in a register and never be reloaded. That asymmetry fits every observation.

* **57** — read `lit[1]` twice through a `volatile` array pointer. `7` = both reads fine
  (refutes consumption); `5` = second read NULL, i.e. **the load consumed the slot**.
* **58** — same for `lit[0]`; the control. If 58 also shows consumption the behaviour is
  uniform and stage 52 only appeared to single out `lit[1]`.
* **59** — read `lit[1]` once, then bounded-walk it. Expect `5` ("rtrim"); `0xB2` on overrun.
  Separates "slot consumed" from "walk broken" for the same element.

### THE REGRESSION TO CLEAR FIRST — a SHA5 wedge that is probably self-inflicted

`wd55` wedged at **SHA5**. Per `sbi_capstone.c:73-76`, "SHA5 then silence" means the hang is
in the **DOMAIN's region-share entry, not the monitor** — i.e. entry-glue territory, BEFORE
cap-init, which is exactly where the one confirmed root cause (an unaligned 8-byte `ld`)
lived.

Cause almost certainly ours: `stage` is a function PARAMETER and the probes build at `-O0`,
so nothing folds and **every staged block's arrays land in every probe binary**. Adding
stages 54-59 silently grew `wd51` from 2 literal arrays to 4, changing the glue's blob-copy
workload for the very domain used as a control. Initialised statics force a blob copy.

**Fixed by `#if CAPSTONE_SQLITE_STAGE == ...` guards around each staged block** so a build
contains only its own arrays. Re-verify a guarded `wd51` returns `0xB1` before trusting any
54-59 result.

### Run order for the next board load (controls FIRST, wedger LAST)

    wd51 (expect 0xB1)  wd53 (0xDF)  wd57 (7)  wd58 (7)  wd59 (5)  wd54 (0xDF)  wd56 (6)  wd55 (6)

`wd55`/`wd56`'s expected value of **6 is proved, not assumed**, so a wrong delta is a
silicon-execution finding and a correct one implicates the walk.

### Traps that bit again TODAY — read before touching the board

1. **Never read `board-<tag>.log` for results.** It carries the accumulated console
   scrollback; grepping it returned markers for stages 30..53 from earlier runs and none from
   the run just performed. **Only `PROBE_SCOPED_OUT` is valid.** The tell was markers for
   stages that were not even in `SQLITE_STAGE_DOMS`.
2. **Prune and ordering must be decided together.** Shrinking the initramfs removed
   `wd51/52/53`, so the next load had NO control; when its only domain wedged there was no
   way to separate "this probe wedges" from "everything wedges now".
3. **Prune only your OWN staged domains.** Never `rm` package-installed ones (`fib`, `sbi`,
   `smode`, `thread`) or anything else inside `build/target/` — that desyncs buildroot's
   stamp files and cost six consecutive boot failures. Keep `sqlite_silicon.dom` and
   `sqlite_host.user` (the freshness gate's reference artifacts).
4. **Image size:** 10.5 MB and 12.1 MB boot; 26 MB and 46 MB do not. 18.6 MB is untested as
   of this writing.
5. Use `build-stage-probes.sh` to build probe batches — it prints per-artifact hashes and a
   distinct-hash count, so a silently-cached build cannot pass as a fresh one.

### Board/infra note

The console tunnel went down mid-session: DNS resolved and TCP :443 connected instantly
while the **TLS handshake timed out**. That presents as a boot failure in the runner's
output; it is not one. `fpga_console.connect()` was also leaking the secret path token into
every captured log — fixed, and the token scrubbed from 75 existing files.
