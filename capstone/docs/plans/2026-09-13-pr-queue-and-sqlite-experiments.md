# Plan: finish the collaborator's PR queue, then the SQLite silicon experiments that the landed work now enables

## Context

**Where things stand (2026-09-13, evening).** Ten of the eleven collaborator PRs are landed on `dev`
as plain merges, each with a control run first; buildroot #2/#3 is also on the FPGA image and proven
by boot sw71 (control `retval=4` with the rebuilt controller; a declaring SQLite image runs to the
size-1 oracle on the new module). S-15 is closed end to end (sw68: the fixed image runs size-20 at
ratio 1.194; sw69: the trap word reads back as mcause 27). The toolchain is freshly rebuilt with #14
(the S-14 spill fix), the QEMU rootfs and the FPGA image both carry the #3 module, and the board's
two hosts are rebuilt for the grown ioctl struct.

**What that landing changed for every experiment that follows.** The ioctl struct grew (#2/#3), and
its size is part of the command number, so any host built against the old struct fails loudly on
both QEMU and the board. Hosts that link `libcapstone.c` are fixed by a rebuild; freestanding
controllers carry a private copy and must be grown by hand (`ladder_perf_ctl.c` is done; eight are
not). Every -O0 image's bytes also changed with #14 (13 truncated capability spills in a
high-pressure cap_init became 16-byte `stc`), so the QEMU icount bridge pair must be re-measured
before any board ratio is compared across the compiler change.

**Two things the user asked to keep in view:** the PR queue (what remains: capstone-qemu #3, held
for a rebase; plus any PR that has arrived since), and the SQLite experiments (the pre-registered
and owed items in §7o / ISSUES.md / the state docs, listed below from the docs, not from memory).

**Decided by the lead on 2026-09-13, before this plan was finalised:** all seventeen new PRs are in
scope; the micropython stack is merged as-is; the size-100 run goes as one overnight boot; the paper
gets a numbers-checker audit and a drafted proposal (no edit without a further go-ahead).

**Sequencing that makes the board and the reviewer's time overlap:** the board items (2.2's
control, 2.x's controls, the size-100 pair) run unattended once staged; the PR reviews and the
zero-board items (2.1, the heap sweep, the paper audit) fill the hours the board is busy.

## Execution order (the board runs unattended; reviews fill its hours)

1. **Minutes, no board:** the housekeeping edits (2.0); B3 (rebuild the three stale overlay hosts,
   same-program gate, QEMU pair); B8 (`toolchain-fresh`); the 2.x host change.
2. **QEMU, ~1 h:** 2.4's pre-run — the fix image `f795151f` under `-icount` at `--size 100` and
   `--size 20` (measured domain count, CPI, oracle match); 2.x's negative control (fix image + new
   host: no trap line, oracle hash); 2.1's static diff, LOAD `cmp`, ① environment control, ④ re-measure.
3. **Board, one overnight boot (2.4 + 2.2 + 2.x's positive control):** driver written with every
   pre-registration in its header, `BUDGET=43200`, entry watchdog on; arms in this order — control,
   `bigregion 1419584`, [pair image + new host `--size 1`: the share-trap positive control],
   baseline `--size 100`, control, **domain `--size 100` LAST**, trailing control only if it
   returns. Launched with a bare launcher; markers read only after this run's own `load_image`.
4. **While it runs:** the PR queue (§1) — standalone, then nginx in order, then micropython as-is,
   then close the #14/#18 rebases; each with its gate run and its result line in the merge message.
   Then the paper audit and the drafted proposal (§3). Then B2 (grow the eight hosts) and the ⑤ᴳ heap
   sweep (2.5) as time allows.
5. **When the boot returns:** invalidators checked before any ratio is written; §7p, ISSUES, state
   docs, the history note; one push per stable point.

## 1. PRs — the queue is larger than it was: seventeen new ones

**Status by repo (read-only `ls-remote` + ancestry; GitHub open/closed state is UNRESOLVED because
`gh` is unauthenticated — the lead may want to set `GITHUB_TOKEN`, but nothing below needs it).**

* **llvm-capstone `dev`:** #11–#13, #15–#17 landed; **#14 and #18 landed by content** (merges
  `d616ea4e`, `8c773a62`) but both branches were force-pushed afterwards as pure rebases
  (identical patch-ids) — **close them, do not re-merge**: #14's new head would conflict with our
  follow-up `bc982fe3` on the same lit test, and neither answers a hand-off ask. **#2 and #3
  (2026-08-22)** are on no branch and not on dev — abandoned or forgotten; the lead says which.
* **caplifive-buildroot:** #1–#3 landed; no new PRs.
* **capstone-qemu:** #2 landed; **#3 still held** — head unchanged, no rebased branch under any
  name; the storewatch stays a hand-off.

**New llvm PRs #19–#35, all from the collaborator, none on dev, all sixteen readable ones clean of
names/emails/trailers in added lines (#35 not yet fetched — the 20-minute background fetch will bring
it; review it then).** Three groups:

| group | PRs, in dependency order | payload | base |
|---|---|---|---|
| standalone | #20 (port-effort counting rule, a tool that refuses an unclassified hunk), #21 (PostgreSQL port effort counted: "96 lines shorter, refutes H4"), #28 (gp-initdesc gate finds the toolchain the standard way) | +311 / +49 / +227 | dev |
| nginx port | #29 → #30 → #31 → #32 → #33 → #34 → #35 (pool in a domain: 78 checks; level-below refuses where it faulted; Sublet port; use-after-destroy demo; alignment + coverage; trace format; replay) | +2,104 through #34, 20 files | dev |
| micropython | #24 → #25 → (branch `micropython/4`, no PR) → #26 → #19 → #22 → #23 → #27 (test selection, score targets, weakref measured, selection residue, GC counters, heap under Sublet, no-coalescing probe) | ~+1,100 incremental | **the rebased #18 head, not dev** — each replays three commits already on dev |

**Order of landing:** standalone first (small, clean, dev-based), then the nginx stack in order,
then the micropython stack as-is (the lead's call), then close the #14/#18 rebases.
Every landing keeps today's discipline: plain merge of commits already on `origin/*`, scan over
the push range by absolute path, `git show` read in full, and **a control run before the pass is
believed** — for a stack that ships a gate (nginx "78 checks", micropython's `run-micropython-gate.sh`
at its new "full level, 559 tests, PASS=549 FAIL=1 FAULT=4 SKIP=5" claim) the gate runs here and
its result line goes in the merge message, whatever it is; for a PR that claims a fault fires
(#32's "plain arm reads its byte, the port faults") both arms run. Anything the PR says that the
run does not show is recorded as the collaborator's claim, not ours.

**The micropython stack's base — decided by the lead (2026-09-13): merge as-is.** git resolves the
replayed #18 changes as identical, so each merge is clean; `dev`'s history will carry #18's three
commits twice under different hashes, and every merge message in that stack says so. If a merge in
the stack does NOT resolve clean (the replayed commits touching a file our follow-ups changed), stop
and resolve toward dev's version by hand, never toward the replay.

**What the new PRs do NOT do:** none answers a hand-off ask (qemu rebase; `-ENOTTY` and accepting
the old `DOM_CREATE` size; the header's skew comment; #14's lit-test wording; #18's 160-test log —
the new stack reports a different level and set). Those stay in the note.

**Reviews apply today's lessons:** a claimed number is reproduced or labelled; a gate is shown to
fire on a negative case before its pass counts; a host that creates domains is checked for which
ioctl struct it carries (the nginx port's host, if freestanding, has the private-struct problem from
day one — check before its first QEMU run, or its "78 checks" will fail loudly on the #3 module and
read as a port defect).

## 2. SQLite experiments — ordered by evidence per board-minute; every claim was checked against its primary source, and an adversarial review corrected five of them (marked)

**Housekeeping first (no board): retire stale "open" bullets.** `current-next-step.md:75`, `:78`
and `current-state.md:26` were discharged by sw68/sw69 (§7o `:3090-3116`); the `cma=`/QEMU-pass/
preflight-C16 bullet (`:93-95`) is DONE (`run-speedtest1-measure.sh:206-222`, `:287-297`;
`preflight-board-run.sh:513-538`); `:547-549` ("the `speedtest1` apparatus is on no remote") is
stale (0 commits off `origin/dev`); `speedtest1-geometry.sh:13-14`'s rationale ("1 MiB because
2 MiB demands order 11") is stale under the one-region rule (1 MiB demands order 11 too for the
static-heap carve). Rewrite those lines so the open list is the open list.

### 2.1 Re-measure the QEMU icount pair on the #14 compiler — zero board time; an owed loop

The pair every -O0 comparison across the compiler change rests on; it was argued unaffected, not
measured. It is the allocator matrix's ④/①, not §7m's silicon bridge: domain memsys5, lookaside
off, vs native `speedtest1_baseline warm`, `--testset main --size 1 --verify`, `-icount shift=0`,
`HEAP 2,097,152` on both. **Corrected by the review: the matrix is the off/off program** — its
oracle is `112006 38bb59fd` (④'s log under `~/capstone-artifacts/matrix/cell4_memsys5/`,
`SPEEDTEST1-CYCLES 692983497`), not the FULL/FLOAT `111130 1e792c9d`; pre-registering the latter
would have compared a different program.

*Why it does not build today, and the knob.* The default static-heap build declares carve
2,316,880 + stack 1,048,576 = 3,365,456, order 11 under the #3 module's one-region rule, correctly
refused (`build-sqlite-silicon.sh:3063-3084`). Only ④ and ⑤ declare that; ⑤ᴳ (2,178,320) and ⑥
(1,268,832) already load on #3 (`readelf` on the archived images). Set `SPEEDTEST1_STACK=385024`
for ④/⑤: need = 1,483,656 + 8,192 + 2,316,880 + 385,024 = 4,193,752 B = 1024 pages = order 10.
**The declared stack is diagnostics only** (`domreq.S:22-25`; the module prints it,
`capstone.c:157-158`, and sizes from `domreq_data`), so the declared image gets the same order-10
block, the same dom_data and the same initial sp the archived run had under the old rule (392,224 B
of stack) — that, not "non-alloc", is the one-variable argument. Prove it: `cmp` the LOAD ranges of
a build with and without the knob (the build's gate at `:3068-3082` compares LOAD *headers*, not
bytes).

*The environment is not one variable, so run the environment control.* Since ④'s run the module
consumes declarations, the rootfs changed, and the QEMU pin moved (§7l `:2650-2655`); the archived
④ cannot re-run (refused). The control is the **native ① re-run**: it must reproduce its archived
count to icount's repeatability (1.3e-7, §7l `:2665-2669`). Reconcile first which ① count is
canonical — the doc's table says 545,623,496 (`:1015`), the archived log 545,609,572, 2.6e-5 apart;
UNRESOLVED which run the row came from.

*Pre-registration, corrected.* `sd`→`stc` / `ld`→`ldc` are 1:1 replacements, so the instruction
COUNT is predicted **identical within icount jitter (~90 on 693 M)** — a 0.01 % band would be
1000× the instrument and absorb any real effect. Do the static diff first: N changed positions,
all inside `__capstone_cap_init`; **if N = 0 the compiler change never reached this image and the
re-measure measures nothing** — say so and stop. Record the post-#14 row in §4g with both hashes;
then rebuild ⑤ with the same knob so the two order-11 cells load on #3 again.

*Adjustment found in execution (2026-09-13):* the #14 whole-image diff is exactly 17 instructions,
all in `__capstone_cap_init`, which runs once (the claim-auditor's finding, `ISSUES.md` S-14), so the
steady-state icount is provably unmoved to within those 17 on a 693 M count — the static argument
alone answers "did #14 move the ratio". The declared-stack knob was confirmed byte-neutral (a
default build with and without `SPEEDTEST1_STACK=385024` produced byte-identical LOAD ranges). A full
④/① re-measurement is therefore optional, and it needs the matrix cell-4 build recipe (memsys5,
lookaside off, the static heap giving carve 2,316,880), which the archived provenance does not
record — the default `build-sqlite-silicon.sh` builds a different, smaller-carve program
(declares 2,496,432). Recover the recipe from the matrix driver before attempting it; do not
substitute the default build.

*Done (2026-09-14):* the recipe is `run-speedtest1-measure.sh` with `SPEEDTEST1_HEAP=2097152` (the
smaller-carve program was a direct `build-sqlite-silicon.sh` call without the measure script's
geometry), plus `SPEEDTEST1_STACK=385024`. Rebuilt on the #14 toolchain: oracle hash, identical LOAD
headers, **692,983,497 = the archived ④ exactly**. ① environment control: −5,993 (1.1e-5), the same
absolute noise as §7l's size-20 repeat. §4g carries both.

### 2.2 R-33's rounding-log positive control — image-free, seconds; prerequisite B3

The module's "not representable" line (`capstone.c:279`) has never fired (§7m `:2822-2827`), so
every zero-rounding reading (sw63/66/68/71) is uninformative. **Corrected by the review: it cannot
ride as a `--pool` arm** — the REGION_ARENA host refuses `--pool/--arena/--tables`
(`sqlite_host.c:344-352`), and a second SQLite `.dom` collides at entry VA `0x10000` (preflight
C15). Use `bigregion.user 1419584` (`bigregion_host.c:52-57`; image-free; counts its own dmesg) —
a stale old-struct host, so **B3 (rebuild it) comes first**. Read the line in the capture channel
(`boot.txt`) *and* as bigregion's count: §7m's doubt is precisely the channel. Land B10 (the
representability gate) *after* this arm — it would refuse 1,419,584.

### 2.3 The S-15 instrument fix — see 2.x, scoped honestly

### 2.4 `main --size 100` — GO (the lead, 2026-09-13): one overnight boot, with three corrections

**Cost, to be corrected in the record:** sw68 measured the CPIs the estimate waited on (native
3.831, domain 3.646, from §7l's size-20 counts `:2645`); on §7l's size-100 baseline count
90,025,541,852 (`:2682`) and its projected domain ratio 1.2467 (`:2707`): baseline 3.8 h, domain
4.6 h, pair 8.4 h + one boot. Write it into the docs whether or not the boot runs.

**Correction 1 — the domain arm at size 100 has never run anywhere.** §7l verified only the
*baseline* at 100 (`:2685-2686`), and preflight C16 keys on the image sha256 alone (`args=` is
displayed, not compared), so a size-1 pass licenses the size-100 boot unchecked. **Before the boot:
run the fix image `f795151f3ed4883b` under QEMU `-icount` at `--size 100` (~20 min) and `--size 20`
(~3 min).** That yields the measured domain count (the ratio was projected), a same-image CPI for
the size-20 cross-check, and the oracle match `23674002 573a4409` — before 4.6 h of board time
depends on it.

**Correction 2 — stage the image by hash, do not rebuild.** `f795151f` was built on the pre-#14
toolchain; a `dev` rebuild today is a different image. One image serves sizes 1/20/100 (`--size`
is host argv). Under the #3 module it declares 1,281,952 → order 10, the same 4 MiB block the old
rule gave — geometry unchanged; let `domdata-budget.py` print it into the driver log.

**Correction 3 — budget and invalidators.** `BUDGET=43200` on both clocks (§7l's upper bound: at
2× CPI the domain arm is 9.1 h; 20,000 s covers only 1.22× the projection). Pre-register in the
driver header: both hashes equal the size-100 oracle; ratio ≈ **1.19** at two decimals (1.2467 ×
3.646/3.831; band 1.14–1.24); `DROPPED 0`; **NOMEM as the named hours-in failure** (arena 128 MiB
built vs 120 MiB measured need, 6.7 % margin, `speedtest1-on-silicon.md:1150-1153` — reads as
`RC≠0`/hash mismatch at the end and costs the whole arm); invalidator #3 **re-registered**: the
count is ~164× size-1, not 100× (§7l measured 164.05 — as written it voids a correct run), checked
against the native arm's board instret; #5 (`capinit-reload-scan`) satisfied by staging the scanned
image, not by re-scanning. Cycles only (`:1201-1207`; the instret image is a separate ~3 h boot).

**Ordering and the entry gate.** Arms: control, `bigregion 1419584` (2.2), baseline, control,
**domain LAST** (it can wedge silently for the whole idle budget), trailing control only if it
returns. Entry watchdog on (UART-line liveness, abort on SHA5-without-SHA6). Its *live* positive
control is still owed (sw64's image `23da3b126a304585` stalls deterministically at share3) but must
not precede a pair unless abort-without-reset is proven — a boot of its own, after.

### 2.5 Checks that may unblock a measurement — corrected

* **⑤ᴳ (geometry-matched Sublet cell) is NOT an S-14 case** (review): its cause 24 is *after*
  `SQ: G/enter` at `HEAP 910,008`, below memsys5's ≥1.5 MiB minimum for `main --size 1` (§4g
  `:1080-1084`); the abort path reads as a fault under emulation, and a post-#14 rebuild predicts
  the same fault. The discriminator is a **heap sweep** on QEMU — `SPEEDTEST1_HEAP` between 910,008
  and 1.5 MiB until ⑤ᴳ completes; if the geometry can be matched, the ⑥/⑤ pair becomes a
  measurement of the discipline itself. Zero board.
* **`json` on silicon**: both exclusions gone, runs on QEMU; the lead ruled depth over breadth
  (`current-next-step.md:375-378`) — revisit only if asked; it is a second `.dom` per boot (C15).
* **R-30 denominator pair** (two region sizes, seconds): after B2 — its probe host is one of the
  old-struct eight.
* **`app`**: write-up (porting-cost result); **`trigger`**: upstream defect.

### 2.6 Tooling — with the prerequisite edges the review found

| item | what | do | feeds |
|---|---|---|---|
| B3 | stale overlay hosts `rtpc`, `bigregion.user`, `sqlite_host_rr.user` (libcapstone-linked) | rebuild against the merged libcapstone; same-program gate; QEMU pair | **2.2, 2.4** |
| B2 | eight freestanding hosts with the old private ioctl struct | grow all in ONE commit (the same two zeroed fields); prove one on QEMU as `ladder_perf_ctl` was; record each instrument's hash change beside its measurement row — recommended, provenance noted | R-30 pair, any probe boot |
| B5 | size-100 timeouts | `BUDGET=43200` per driver (defaults 90 s / 30 s) | 2.4 |
| B10 | arena gate does not check representability | add (power of two or granule multiple); control: 1,419,584 must fail it — **land after 2.2** | — |
| B8 | `toolchain-fresh` after the #14 rebuild | run once, record | — |
| B4 | `capinit-scan.py` has no positive control | record as unproven; the `sd ra` spill count is the gate of record | — |
| B6 | `domdata-budget.py` MAX_ORDER off by one, right by coincidence | fix; control: synthetic `CONFIG_ARCH_FORCE_MAX_ORDER=10` → ceiling 10 | — |

### 2.x The S-15 instrument fix — host-side, low-risk, and honestly scoped

The interp glue's `.Ldomain_trap` (start-gp-captable-interp.S:959-993) packs `0xF | mcause<<22 |
(mepc-_start)>>2` (`0xE` for the `INTERP_TRAP_A4` variant, `:1023-1026`) into the `res` slot saved
at entry. For a CALL entry that is the monitor's `res`, read by the host as `obs=`; for a SHARE entry
it is the shared region's first word, which nothing reads. **Corrected by the review: this protects
diagnostic (trap-vector) images only.** The readback runs after the share *returns*; a measurement
image has `mtvec = 0`, so its fault re-faults forever and the share never returns — no readback
executes. sw64/sw66's eight hours are bounded by the *entry watchdog*, not by this. What the
readback buys is narrower but real: on a trap-vector image the trap is named at the share instead
of surfacing later as a misleading `obs` (sw65/sw67's `0x5117BAD3` cost an audit cycle). It diverges
by platform for LINEAR (`REV_BORROWED`) shares — Sublet's `--arena` — where on silicon `ldc` clears
the source slot and the handler re-faults; noted, not relied on there.

Change: in `sqlite_host.c`, after every share read word 0 of the region; for share1 compare it to
the `phase` value the host wrote rather than nibble-testing (the domain's share handler writes
nothing into any region, `speedtest1_measure.c:695-735`); a `0xF`/`0xE` top nibble prints
`SQ: share-trap=<word> mcause=<n> off=<off>` and stops the run. Pre-write a non-`0xF/0xE` sentinel
into the arena before share3 so "handler never ran" is a positive reading, not a zero. Default-on.

Controls — **the positive control is board-only** (QEMU cannot raise the delin fault, §7o
`:3043`): the pair image `214b300efd169f03` with the **new** host at `--size 1` must print
`share-trap=0xF6C09D13 mcause=27 off=0x9D13` at share3 and nothing at share1/2, then
`obs=0x5117BAD3` (sw69's reading; sw69's host is a different binary and licenses nothing for this
one). Negative: the fix image `f795151f3ed4883b` with the new host prints no line **and** hashes to
the oracle — judged on the hash, so the readback's mmap is shown not to perturb the run. Both arms
ride in the size-100 boot's preamble if that host is the one staged; otherwise a boot of their own.

The proper fix is monitor-side and smaller than first described: the share path already forwards
its return through the module (`capstone.c:485-500`); libcapstone's `void
shared_region_annotated()` drops it (`libcapstone.c:583`). Monitor reads word 0 through its retained
NONLIN copy after `SHA6` and returns it; the wrapper stops dropping it; the host prints it. No
module or domain ABI change, no 128 MiB host mmap. Hand-off, described that way.

## 3. Paper — a proposal, not an edit (ask-first rule; Overleaf owns the remote)

**Finding (verified in `capstone/paper/parts/evaluation.tex`, tree clean at `2bc09e1`).** The paper
carries no measured speedtest1 ratio at all. SQLite is priced by an *estimate*: borrows × 171 cycles
over instr × CPI, "~1%" (`:649`), "≤6%" (`:628`), and the text keeps CPI = 1 "because the measured
CPI comes from those kernels rather than from SQLite itself" (`:660-661`). That rationale is now
stale: SQLite's own native CPI is measured on silicon (`main` 3.769, measurements doc `:962`;
baseline 3.76 / domain 3.81, `:1158`). And the quantity the estimate approximates is now measured
directly, and absent from the paper:

| measured whole-benchmark ratio | value | where |
|---|---|---|
| `main --size 20`, silicon, boot sw68 | 1.194 (54,214,856,567 → 64,732,455,367 cycles) | §7o `:3105-3110` |
| `main --size 1`, silicon pair, post-flash | 1.2195 | §7m `:2795` |
| seven testsets, silicon | 1.196 | §7k `:2371` |
| allocator matrix, QEMU `-icount`, memsys5 | 1.2703 | §4g `:971` |

**What to propose to the lead (one decision, three parts):** (a) add the measured whole-SQLite
ratio (the size-20 row as the headline, size-1 beside it) next to the estimate, or in its place;
(b) replace the CPI = 1 sentence with the measured SQLite CPI; (c) carry the two caveats the
measurements doc binds to every silicon number — the bitstream does not meet timing (`:2489`; prefer
instruction counts where a claim can be carried by either, `:2586-2588`) and every §7 speedtest1 row
was measured with SQLite's lookaside pool OFF, a configuration SQLite does not ship (`:2593`,
`:2620-2628`). The framing choice the doc explicitly leaves to the paper — user-work-vs-user-work
denominators (`main` 1.294 tick-adjusted) versus whole-machine (`:2447`) — goes in the same question.

**Before any wording:** run the `paper-numbers-checker` agent (read-only; roster in
`docs/ref/SUBAGENTS.md`, definition `.claude/agents/paper-numbers-checker.md`) with two inputs it
cannot infer: the live section is `parts/evaluation.tex`, not `old-parts/` (the measurements doc
still points at the old path, `:150`), and `old-parts/` + `proposals/` are scratch `\input`s
(`main.tex:119-133`) to be excluded. Its "measured but not in paper" list is the checklist for the
proposal. Two unmerged remote branches of the paper repo may carry newer evaluation text; check them
before proposing against `main`.

Nothing in the paper is contradicted by S-15 or by the PR landings; the lag is by omission. No
"flagship 3.2%" withdrawal exists to act on (the `beebs_prime` 1.032→1.683 correction is already in
the paper at `:497,552`).

## 4. Hand-offs (to the collaborator through the lead; the note under `/tmp/capstone/`, no names)

Already written into `/tmp/capstone/pr-review-notes-for-the-collaborator-2026-09-13.md`; the plan only
keeps them tracked against any new PR that answers them:
* capstone-qemu #3: rebase onto `c128-qemu-merge` dropping the two commits already landed in
  superset form (`c64867389e`, `b59d116983`); the storewatch (~104 lines) is the net-new content.
* buildroot module: an unrecognised ioctl should return `-ENOTTY`, not 0 (the loader then hands back
  `dom_id -1` with no kernel error); accept the OLD `DOM_CREATE` size too, treating absent fields as
  zero, so the "appended, simply dropped" header claim becomes true; when a declared block exceeds
  the order ceiling, fall back to the historical rule with a `pr_warn` rather than refuse, or land the
  two-region path; compute the split slack from the granule (8 KiB is short once `repr_len ≥ 4 MiB`).
* llvm #14: the lit test does not gate the defect (byte-identical pre/post-fix); capinit-scan
  under-detects the spill form; the gate of record is the `sd ra` spill count on a pressured image.
* llvm #18: the 160-test `PASS=100/FAIL=60` figure still owes its log or a "measured with #2+#3"
  qualifier (the landed gate is the 4-test smoke).
* the share-entry trap: the proper fix is monitor + wrapper (the monitor reads the region's first
  word through its retained copy after `SHA6` and returns it; libcapstone's `void` wrapper stops
  dropping the return); no module or domain ABI change. This plan lands the host-side readback as
  the instrument in the meantime, scoped to trap-vector images.

## Verification

* **Every board arm is judged on its pre-registered reading, written in the driver header before
  the boot, with the control first and the run VOID if the control fails** (`retval=4`). Results are
  cited by image hash from the run-scoped `boot.txt`, never from the raw log (the console replays the
  previous boot on connect — it bit twice today).
* **Every rebuilt host passes the same-program gate before staging**: builder found by the output
  filename the driver invokes (`ladder_perf_ctl`, not the slot name), the loader's marker strings
  (`/dev/capstone`, `RESULT`, `retval=`) present, size within 2× of the previous binary, and the
  QEMU pair (old fails loudly, new prints `RESULT k800 retval=4`) run against the #3 rootfs first.
* **2.1**: the rebuilt ④ must hash to the off/off oracle `112006 38bb59fd` and report
  `HEAP 2,097,152`; the LOAD ranges with and without the stack knob must `cmp` equal; the native ①
  re-run must reproduce its archived count to ~1e-7 (environment control); the static diff must show
  N > 0 changed positions in `__capstone_cap_init`; only then is ④'s count compared to 692,983,497,
  predicted identical within icount jitter.
* **2.2**: the rounding line must appear exactly once on the `bigregion 1419584` arm, in the
  capture channel and in bigregion's own count, and zero times on every other arm of the same boot —
  positive and negative control in one boot.
* **2.x**: on the BOARD (QEMU cannot raise the fault): the pair image `214b300efd169f03` with the
  new host prints `share-trap=0xF6C09D13 mcause=27` at share3; the fix image with the same host prints
  nothing and hashes to the oracle. A boot where neither prints has an instrument that cannot fire.
* **2.4**: the QEMU `-icount` size-100 run of `f795151f` matches the oracle `23674002 573a4409`
  BEFORE the boot; on the board: same boot, arena 128 MiB, both hashes equal that oracle,
  `DROPPED 0`, ratio inside 1.14–1.24 at two decimals, NOMEM and the re-registered invalidator #3
  checked before the ratio is written, cited by hash from `boot.txt`.
  *Done 2026-09-14 (boot sw73): both hashes `23674002 573a4409`, `DROPPED 0`, ratio 1.1819 → 1.18,
  all five invalidators checked; §7p.*
* **Landings**: every merge is a plain merge of commits already on `origin/*`, scanned by absolute
  path over `<branch> --not --remotes` and gated on the scan's exit status; `git show` read in full,
  not `--stat`; submodule pushed before the parent references it; no `Co-Authored-By`; no names.
* **Paper**: nothing is edited without the lead's go-ahead; the checker's report is the input to the
  question, not a licence.

## Documents to update (the last step of every item above, committed with it)

* `docs/ref/fpga-silicon-measurements-for-paper.md`: §4g gains the post-#14 icount row (2.1); §7m's
  R-33 caveat is closed by the bigregion rounding control (2.2); a new §7p for the size-100 pair if it runs
  (2.4), carrying the corrected cost derivation and the five invalidators' verdicts; the instrument
  fix (2.x) recorded where §7o says "owed".
* `docs/ref/ISSUES.md`: S-15's "Instrument fix owed" line closed by 2.x (with its scope: trap-vector
  images); R-33's rounding-log control (the bigregion arm) and, if run, the bottom-truncation arm; M-1's policy question (should measurement images carry the
  trap vector by default now that sw69 shows the handler is reachable after a real fault) stated as
  a question for the lead, not decided.
* `docs/state/current-next-step.md` / `current-state.md`: the three stale bullets and the stale
  "on no remote" line retired; the size-100 cost corrected (8.4 h, DERIVED, inputs cited); the PR
  table's final state; the eight private-struct hosts' status after B2.
* A dated `docs/history/` note for the #3 landing (sw70 void → sw71 pass, the two fixes, the gate),
  since it is a root-caused mistake with a process lesson, not a design decision.
* The hand-off note under `/tmp/capstone/` (never committed) updated with anything a new PR answers.
