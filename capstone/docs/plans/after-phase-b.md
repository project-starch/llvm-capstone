# After Phase B: the closing boot, R-26 follow-through, Q-06 localisation, a load guard for the nightly

**Status (2026-09-09).** Approved by the lead 2026-09-09. Step 1 DONE (boot sw38, fw_payload
b1680288381d: 9/9 at the oracles, one HOLE line, zero fault tags). Step 3 DONE as far as it goes
(ISSUES.md Q-06: the fault is `sbi.dom`'s `query_region` reading an untagged value back from a CPMP
slot; two candidate mechanisms recorded, owner unchanged). Step 4 DONE (`run-nightly.sh` records
other users' CPU per suite and marks suites that ran under load; three controls). Step 2 is with
the RTL lane (fix + switcher test + synthesis); this lane's `fence.i` drop waits for the fixed
bitstream on the board.

## Context

Phase B is done except the kernel (deferred by the lead). Four things follow from it, each independent:

1. Every Phase B boot ran on an evolving firmware; the final one (`fw_payload 5dd1c265a70d`, monitor
   5b27d01) has only ever run the k800 control and the transfer probe. One boot that runs the whole
   validated set on it closes Phase B on the firmware that actually ships.
2. R-26 is now **measured in simulation** (RTL lane, `docs/history/08-09-2026_22-00-00_r26-…md`): a
   younger load checks CPMP against the entry a pending `CCSRRW` is about to replace. The monitor keeps
   four `fence.i` on that account (`sbi_capstone.S:90,176,189` after CCSRRW; `sbi_capstone_dom.c:47`,
   the UART-mint one, kept untested). The RTL fix and the remaining directed test are the RTL lane's;
   dropping the fences is this lane's, and only after the fixed bitstream is on the board.
3. Q-06 (`ISSUES.md:956`): the null-blk split domain's S-mode init faults on an untagged `lcc`
   (cause 24, QEMU) yet the device serves I/O. Pre-existing (A/B against the pre-unification `sbi.dom`
   identical). Nobody has localised it; it is board-free and bisectable.
4. The 5A nightly tier came back 17/18 with six infra flakes under another user's MySQL/Bazel load
   (one per tier earlier that day). The tier cannot say "I ran under load", so a flaky tier reads
   like a regression until someone reruns by hand.

## 1. Closing boot sw38 on the final firmware (this lane, ~40 min, no unknown)

**Stage set (10 domains, all previously at their oracles on earlier Phase B firmware):** k800, the
six BEEBS rungs, SLT `select1`, the transfer probe. **M-2 is left out**: `m2.dom` and `sqslt1m.dom`
both link at entry VA `0x10000` (preflight C15 blocks the pair; sw33 already showed the refusal on
silicon and neither the module bound nor the monitor's `create_region` check changed since). Relinking
M-2 elsewhere would need a `DOMAIN_BASE_VA` knob in `build-m2-region-overflow.sh` that it does not
have — not worth adding for a confirmation boot.

Steps (a copy of `~/capstone-artifacts/unify/b5b-board.sh` minus the commit chain, plus the probe):
- Restage `sqslt1m.dom`, `sqlite_host_1m.user`, `select1.test` from `~/capstone-artifacts/overlay-attic`
  into `overlay/test-domains` AND `build/target/test-domains`; leave `m2.dom`/`m2_region_overflow.user`
  in the attic. `A=linux-rebuild` then `A=opensbi-rebuild` (`TARGET=fpga LINUX_PAYLOAD=1`,
  `CAPSTONE_CC_PATH` absolute). Assert the `fw_payload` sha256 is `5dd1c265a70d…` — the monitor and
  its `.c.S` must not change; if the hash differs (OpenSBI timestamp), gate on `.c.S` byte-identity to
  `~/capstone-artifacts/unify/2B-fpga.c.S` and record the new hash in `firmware-with-q03-hole.txt`.
- Stage entries: the seven `|label` rungs as in `b5b-board.sh:22`, then
  `/test-domains/sqslt1m.dom:--slt /test-domains/select1.test` with
  `SQLITE_HOST=/test-domains/sqlite_host_1m.user`, then the probe entry
  `/test-domains/rtpc|revxfer:/test-domains/revxfer.dom` LAST (records `revxfer.oracle`/`.qemu-pass`
  already carry the image hash). `SQLITE_STAGE_TIMEOUT=2400`, `ENTRY_STALL_S=420`. No
  `PREFLIGHT_ALLOW_SLOTS` (the firmware is listed).
- Order: control first, rungs ascending, SLT, probe last (the post-transfer state is the only thing
  that could take the core).
- Read the run's own transcript segment; expected byte-for-byte the sw35 readings for the eight
  (k800 = 4, the six rung oracles, `records=1031 … completed=1`) plus `RESULT revxfer retval=574619742`
  and exactly one `HOLE:` line; zero fault tags.
- Record: nine rows in `tests/board-results/2026-09-05.tsv`, the plan doc's status paragraph ("closed
  on the shipping firmware, boot sw38"), `current-state.md` one line. One parent commit.

## 2. R-26 follow-through (RTL lane first, then this lane)

**RTL lane (their branch `r26-ccsrrw-stale-read`, tests in `verif/tests/custom/capstone/r26-*.S`):**
- Fix: raise `flush_o` in the CCSR write block (`core/csr_regfile.sv:2392-2418`, the `CCSR_CEPC`,
  `CCSR_CSCRATCH`, `CCSR_CPMP*` cases) the way the side-effecting CSR writes do at `:1160-1180`.
  Acceptance: the `ldmiss` arm turns PASS while the positive control and the post-fence control still
  trap; `nodelay`/`div`/`fence` arms unchanged. `bash verif/sim/rtl-lint-gate.sh` PASS, claim-auditor
  on the diff (latches, the `flush_o` cone — it is a controller input, check it is not one of the
  `UNOPTFLAT` loops), then synthesis before anything else — batched with the R-25 fix into ONE
  bitstream.
- The second directed test: the `cscratch`/`cepc` readers in `dom_switch_read_process`
  (`csr_regfile.sv:401-432`) — `CCSRRW … CSCRATCH` followed by a CALL, with an older cache-missing
  load ahead of the CSRRW (`S12_MEM_DELAY=40`), the switcher's restored capability compared against
  the new value. Prediction written down before the run: with the fix, PASS; without, the CALL restores
  the OLD cscratch. This is the shape with the board consequence (`sbi_capstone_init.S:44-51`).
- Registry line (`ISSUES.md` R-26) is this lane's to edit once they report: "fixed in RTL, sim-verified,
  bitstream pending" and then the bitstream hash.

**This lane, only after the fixed bitstream is flashed (the flash itself is ask-first, the lead's):**
- Two firmware-only variant boots with the item-8 machinery (`scratchpad/fence-variant.py` pattern,
  `board-b8.sh`): D = drop the three CCSRRW-adjacent `fence.i` in `sbi_capstone.S`; E = D plus the
  UART-mint one in `sbi_capstone_dom.c:47`. Each: the closing-boot stage set, control first. Expected
  CLEAN 8/8 + probe on the fixed silicon; on the CURRENT bitstream D is predicted to fail
  intermittently (that is not a run to make — sim already answers it).
- If both clean: one monitor commit removing the four with the R-26 history note as the reason;
  `.c.S` gate (FPGA differs only at those sites, QEMU byte-identical), pin bump chain, push.

## 3. Q-06 localisation (this lane, QEMU only, bounded to one session; result handed to the null-blk owner)

Facts on file: `nullb_split.smode.c:181` is a naked `__init` that sets `sp`, then loops:
`REGION_ID_TO_BASE(METADATA_REGION_ID)` → `*(unsigned long *)base` → dispatch → `DOM_RETURN`. The
fault is `lcc rd=x6 rs1=x5 sel=3` at `.init.text` pc `0x1015940xx` with x5 UNTAGGED; the monitor side
is `call_domain_with_cap(arg0..arg3)` (`sbi_capstone.c:1624`). The device still serves I/O, so the fault
is on a path the I/O does not need — or the init is re-entered per call and faults only on one function
code.

- Step 1 (static, 30 min): `llvm-objdump -d` the `nullb_split.smode.ko` `.init.text`; find the faulting
  offset; name the instruction that produces `x5` (a `ldc` from a cap table? a plain `ld` of a
  capability-typed word? the `mv sp` clobber?). Check how `REGION_ID_TO_BASE` expands and whether the
  region-base table the module reads is populated by `call_domain_with_cap` at all (what arg1..arg3
  carry, and where the monitor stores them for the domain).
- Step 2 (dynamic, one QEMU boot per variant, serialized under the lock): early-return variants of the
  init that STOP THE FLOW — v0 `DOM_RETURN` immediately (does entry/return work), v1 read the metadata
  base and return it as `rv` (is the base tagged), v2 read `function_code` and return it. Build each as
  its own `.ko` (the runner takes a path), run under `run-nullblk-split-io.sh`, classify by the returned
  value, not by the absence of the fault line.
- Step 3: write the localisation into the Q-06 entry (mechanism + the instruction + what the monitor
  hands over), owner unchanged. No fix in this lane unless it is a one-line monitor handoff bug, in
  which case: fix, `run-nullblk-all.sh` green, own commit.
- Not board work; no paper impact.

## 4. Nightly-under-load guard (this lane, small, own commit)

`capstone/tests/run-nightly.sh` (`JOBS` at `:124`, suites listed at `:51-75`, per-suite loop writes the
`[suite] … -> PASS/FAIL` lines and `report.md`):
- Sample before each suite: 1-min load from `/proc/loadavg`, and the top foreign CPU consumers
  (`ps -eo user,pcpu,comm --sort=-pcpu`, users other than `$USER`, summed `pcpu`). Write both into the
  report row (`| suite | result | duration | load | foreign-cpu | log |`) and into the log line.
- Threshold: if load1 > 0.5 × nproc or foreign CPU > 400 % at suite start, mark the suite row
  `UNDER-LOAD` and, on a FAIL, print `FAIL (under load — rerun the case alone before reading it as a
  regression)`; the exit status is unchanged (a gate must not soften itself). `--refuse-under-load`
  makes it wait up to 30 min for the load to drop, then proceed and mark; default is mark only.
- Positive control before the commit: run `--only smoke` while a deliberate `stress -c 60` (or a
  `yes > /dev/null` ×60 loop in a `systemd-run` scope) is running — the row must show `UNDER-LOAD`;
  without it, must not. Negative-test the parsing (a `report.md` with the extra columns still parses
  in whatever reads it — grep for readers of `report.md`).
- Do NOT change the BEEBS runner's no-retry policy for guest-phase timeouts (a silent guest before the
  loader's first line is what a wedge also looks like; only a first-in-boot rerun can tell them apart,
  and that stays a human step, as done for sglib-hashtable).

## Order and cost

| step | board | QEMU | who | blocks on |
|---|---|---|---|---|
| 1 closing boot | 1 boot (~40 min) | — | this lane | nothing |
| 4 load guard | — | one smoke under load | this lane | nothing |
| 3 Q-06 | — | ~4 boots | this lane | nothing |
| 2 RTL fix + switcher test + synthesis | — | sim | RTL lane | nothing |
| 2 fence.i drop (D, E) | 2 boots | — | this lane | the fixed bitstream flashed (ask-first) |

Do 1, then 4 while nothing is on the board, then 3; 2's monitor half waits for the RTL lane.

## Verification ("done")

- 1: sw38 rows in the tsv at the sw35/sw37 oracles, zero fault tags, one HOLE line; plan doc + state doc
  say Phase B is closed on `5dd1c265a70d`.
- 2: RTL lane reports `ldmiss` PASS + switcher test PASS + lint PASS + synthesis run; after the flash,
  boots D and E clean; monitor commit with `fence.i` count 152 → 148 in the linked firmware.
- 3: Q-06 entry names the instruction and the handoff; the variant readings are quoted.
- 4: report rows carry load/foreign-cpu; the stress positive control flagged; exit status unchanged.
- Every commit scanned with `precommit-scan.sh`; nested pushes through `/tmp/capstone/push-final.sh`
  where the lead's credential is needed (only if the monitor changes, i.e. step 2's second half).

## Risks

- Step 1 could fail on an infra draw (control fails ~1 in 5): a boot whose control fails is VOID, rerun.
- Step 2's flush may enter a combinational cone that already carries an `UNOPTFLAT` loop — that is the
  highest-risk edit class; synthesis before any board use, no exceptions.
- Step 3 may end at "the monitor hands an untagged base" (a monitor bug on the QEMU line's null-blk
  path) — then it is a fix for this lane, gated by `run-nullblk-all.sh`, not a hand-over.
