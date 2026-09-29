# R-43 — R-35's revocation cache refused LIVE capabilities once ~256 revocation ids were live

**What it is.** R-35's fix (`capstone-ariane 4ad0df694`) lets an M-mode load/store through only when the
capability's exact 30-bit revnode id is resident in a 4-way × 64-set cache and marked live, and it
**denied on a miss**. That is safe (it never allows a revoked capability), but once a program holds more
than about 256 live revocation ids, entries are evicted and a perfectly LIVE capability is refused with
cause 25. Long-lived capabilities (a domain's own globals) are the first casualties.

**Siblings and nested issues, so a reader with a neighbouring symptom is redirected:**
- **R-35** (`../R35-revoked-reference-retains-authority/`) is the defect whose fix caused this.
- **R-42** (`../R42-icache-killed-miss-refill/`) is the bitstream this was CONFIRMED on; it is unrelated
  (an I-cache cycle loss) and only shares the lineage.
- **R-45** — the revocation ORDERING window, found while testing this fix and closed in the same
  bitstream: a load/store issued right after REVOKE/DROP could be checked before the revocation took
  effect. Nested below.
- **R-46** — a commit-stage refetch keeps stale PC-capability metadata. Found by the final audit of this
  fix; accepted for this bitstream by the project lead, closed on the replay path only. Nested below.
- **R-44** is the same optimistic adopt at the CPMP (S/U mode). Not touched here; nested in R-35's folder.

**Record, by date:**
- **2026-09-25:** confirmed on silicon (`caplifive_r42_6cbdaeeb4.bit`, boots r42b3/r42b4/r42b6/r42b7) and in
  RTL simulation (`verif/tests/custom/capstone/r43-evict-live.S`, capstone-ariane `93f509f54`).
- **2026-09-28:** first fix `bbd4d1478` + R-45 `0f5185a6d` (branch `r43-query-on-miss`), every simulation
  arm as predicted: `results/sim-query-on-miss.result-lines.txt`.
- **2026-09-29:** that fix **REFUTED BY SYNTHESIS** — WNS −24.495 against R-42's −10.615, all of the worst
  500 paths through its stall gate: `results/synth-0f5185a6d.result-lines.txt`. Not a flash candidate.
- **2026-09-29:** second fix `8f6a0af98` (replay instead of stall). Shipping build passes every simulation
  arm; lint at baseline; the variant batch and the after-audit run in parallel with synthesis (the lead's
  decision). Synthesis requested as `caplifive_r43_8f6a0af.bit`.

Registry: `docs/ref/ISSUES.md` R-43, R-45, R-46. Plan and pre-registration: `docs/plans/r43-query-on-miss.md`.

## The problem, in one picture

```
   access via capability C (id c)
          |
          v
   +-------------------------------+        4 ways x 64 sets; set = id[5:0]
   | revocation cache (R-35 fix)   |        filled by passive taps on the rev-node's own node traffic
   |   hit, live  -> ALLOW         |
   |   hit, dead  -> DENY 25       |
   |   MISS       -> DENY 25   <---+---- R-43: more than 4 live ids in C's set evict C's entry,
   +-------------------------------+           so a LIVE capability is refused
```

On silicon (`caplifive_r42_6cbdaeeb4.bit`) this killed both R1 harness runs on their first invocation
(boot r42b3: a globals capability allowed at `+0x4f40` and denied at `+0x4ff4`, one `mrev` later), the
live128 and live512 sweeps (cause 25 at the sweep's `lbu`, `tval` an early alias leaf), and P1 cell 6 at
`-O2`. The live16 sweep passed (544/544), because 16 ids fit. The emulator runs every one of them to
completion.

## The first fix, and why synthesis refused it — REFUTED, kept as the trail

`bbd4d1478` resolved a miss by HOLDING the access: a combinational `rvc_stall`, built from the cache
lookup, gated the load/store unit's `valid` while a probe asked the rev-node unit. Correct in simulation
(every arm in `results/sim-query-on-miss.result-lines.txt`), and it put the whole revocation lookup IN
SERIES with the load request:

```
   BEFORE (R-42, 6cbdaeeb4)                       FIRST FIX (0f5185a6d)
   fu_data_q ---> lookup ---> cap_exception         fu_data_q ---> lookup ---> rvc_stall ----+
                                |                                    |                       |
                          [MMU flop] (registered)              cap_exception            ld_valid_i (gated)
                                |                                    |                       |
                        load unit finishes                     [MMU flop]               data_req --> D-cache
                                                                                        grant --> rev-node
                                                                                        read grant --> fill
                                                                                        taps --> scoreboard ...
   worst path: 0 of 500 through the lookup        worst path: 500 of 500 through the lookup
   WNS -10.615                                    WNS -24.495; 99 logic levels, 56.4 ns of ROUTE (congestion)
```

The "ORDER test" — does an `rvc_` cell precede the load unit / MMU on a worst-500 path — reads 0/500 on
R-42 and 500/500 on the first fix, so the cause is the gate and not R-45's flush (no worst path runs
through it). Loops: the same families cut differently, not new ones (an earlier "3 new loops" reading was
withdrawn the same day). Lint and simulation were blind to all of it, as CLAUDE.md says they are.

## The second fix: REPLAY a missed access, never hold the request

`8f6a0af98` keeps everything below the gate — the probe endpoint, the ALLOW/DEAD records, the two-part
timeout, R-45 — and changes only what a miss does:

```
   M-mode load/store, revocation-cache MISS (no hit, no ALLOW/DEAD record)
          |
          v
   LSU: mark the access with an INTERNAL replay cause on the EXISTING registered exception path
        (cap_exception -> MMU flop -> the load/store unit finishes WITHOUT a memory operation)
        and start the probe:
             probe_req(index)  -- a new, lowest-priority, NO-RESPONSE rev-node endpoint
             the rev-node reads node[index]; the cache's EXISTING read tap sees that read
             and reports {generation, index, valid}
          |
          v
   commit: the marked head is NOT an exception (stripped from exception_o; never reaches mcause);
           wait while the LSU's REGISTERED probe-pending flag is set;
           then flush and re-fetch THE SAME pc (frontend: pc_commit + 0, restoring the head's own
           PC-capability metadata, so R-46 cannot fire on the replay)
          |
          v
   the re-executed access:  hit, or ALLOW record (set by a live probe)      -> allowed
                            DEAD record (dead / stale generation / timeout) -> cause 25 (fail closed)
```

What makes it hold, each checked in source by the before-audit:
- **Nothing new enters the load unit's request path.** The only new wire from the LSU to commit is the
  probe-pending flag, a flop. Commit's replay term reads the head and that flop — never `commit_ack` or
  `pc_cap_ex_valid`, whose cones carry the revocation broadcast.
- **The probe outlives the access's pop**: it clears only on resolve, timeout or a flush. Clearing it on
  pop would cancel it and livelock the replay.
- **ALLOW** is set only by a live probe resolution, keyed by the exact 30-bit id, cleared by any
  invalidation of its index (broadcast or a dead write), never by a flush.
- **DEAD** is set by a dead/stale resolution or a timeout, cleared only by a write to its index, never by
  flush or pop. A dead 30-bit id never becomes live again, so a stale DEAD cannot falsely deny.
- **A stale generation denies, never stalls**: the tap installs the node's CURRENT generation, so the
  resolution compares all 30 bits (the `mgen` mutant, which drops that compare, must ALLOW arms 5/6).
- **Younger rev-node operations cannot run twice**: DROP/REVOKE/MREV/SPLIT/DELIN issue only when every
  older instruction has committed, and a marked head is uncommitted.
- **Bound:** at most 3 replays per dynamic access (the trace counts them: 7 replays in the test, at most 1
  per pc). Timeouts fail closed: ~1M cycles before the rev-node accepts the probe (it is served only in
  IDLE, never during a walk, and silicon walks cost up to 11,811 cycles), 65,535 after.

### The refusal record — observation only, batched into the same bitstream

If a live capability is still refused on silicon, nothing today says WHY. So the build carries a sticky
record of the FIRST cause-25 verdict since reset — the 30-bit id and which arm produced it — readable
through the debug-switch mux at values **204..208** (bank `3'b110`, registers `01100..10000`; all
UART-safe, `(value & 3) == 0`):

```
   204 = { 2'b00, arm[3:0], ~v, v }      arm one-hot: bit2 hit-dead, bit3 same-cycle invalidation,
   205 = id[7:0]    206 = id[15:8]                    bit4 probe DEAD, bit5 timeout
   207 = id[23:16]  208 = { p_hi, p_lo, id[29:24] }   p_lo = XOR id[14:0], p_hi = XOR id[29:15]
```

Valid is encoded `{~v, v}` and the arm is one-hot **so that a contaminated read is detectable**: the LED
stretcher ORs apertures on a running core, and a corrupted read shows as `11`, `00` or a non-one-hot arm.
Positive control: the R-35 stale probe (image `35fb3fec`, the last acceptance arm) must latch its own id;
negative: a passing boot reads `10` (empty). In simulation the record latched the test's first denial
(id `0x616`, arm `0001` = hit-dead) and nothing else.

## R-45, nested: the revocation ORDERING window, closed in the same bitstream

```
   REVOKE r            ... walk runs in the rev-node ...          REVOKE commits (after the walk)
   ld x, 0(C)   <-- checked HERE, before C's node is written dead ---^
                    so it was ALLOWED                          R-45 fix: flush younger instructions at
                                                               REVOKE/DROP commit; the ld re-executes
                                                               after the revocation and is DENIED
```

Found by this test's arm 7: nothing held LSU issue behind an in-flight DYN op, so a younger load/store
passed its check against the pre-revocation state. It **predates R-43** — on `6cbdaeeb4` the same read is
allowed whenever the entry is still resident (arm 7r); deny-on-miss had refused the evicted case only by
accident. The R-35 fixture's 64-nop barrier after every REVOKE exists precisely to step around it. Fix
(`0f5185a6d`, carried into `8f6a0af98`): `commit_stage.sv` raises `flush_commit` when a REVOKE or DROP
commits, as R-26 does for a capability CSR write. One pipeline flush per REVOKE/DROP; the D-cache is
untouched. The `noflush` mutant must ALLOW arms 7 and 7r — that is the control proving the flush is what
closes it. Registry: `docs/ref/ISSUES.md` R-45.

## R-46, nested: a commit-stage refetch keeps stale PC-capability metadata (accepted, closed on the replay only)

```
   head:  REVOKE (or AMO / CSR write / fence)          younger: CJALR -> code capability K2
              |                                                    |
              |  commits -> flush_commit, refetch pc_commit + 4    |  resolved in EX before the flush:
              v                                                    |  npc_metadata := K2's metadata
   frontend: npc_d = pc_commit + 4                                 |
             npc_metadata_d = npc_metadata_q   <-- carried forward: it is K2's now, not the head's
              |
              v
   the re-fetched instructions after the REVOKE run under K2's bounds/permissions
   until the next capability control-flow change  (consumers: pc_cap_check in commit, CALL's saved
   caller metadata)

   R-43 v2 replay refetch:  npc_d = pc_commit + 0, npc_metadata_d = the HEAD's own pc_metadata  -> closed
   ordinary refetch (+4):   unchanged                                                              -> open
```

Harmless while a domain has a single code capability, because a `ret` restores the same metadata — the
case in R1, the sweeps and the corpus. With several code capabilities it can raise a spurious bounds fault
or run code under the wrong capability. Pre-existing (every AMO/CSR/fence refetch), made routine by R-45's
flush, which is why the project lead accepted it for this bitstream. The fix candidate is the replay path's
own restore applied to the ordinary refetch. Registry: `docs/ref/ISSUES.md` R-46.

## Evidence

| what | result | file |
|---|---|---|
| RTL sim, first fix + R-45 (2026-09-28): arms 2/2s/5/6/7/7r/8/9a/9b and the mutants tieoff / noflush / mgen / mto / flushproxy / noclear / noclearflush | every arm and every mutant as pre-registered; R-35 fixture 7 traps; 92-test sweep 0 differences; lint PASS (UNUSEDSIGNAL re-baselined 736→737, the probe read's unused return value) | `results/sim-query-on-miss.result-lines.txt` |
| synthesis, first fix (`0f5185a6d`) | **REFUTED**: WNS −24.495, ORDER 500/500, route 56.4 ns on the worst path; R-45 not implicated | `results/synth-0f5185a6d.result-lines.txt` |
| RTL sim, second fix (`8f6a0af98`, shipping build, 2026-09-29) | every arm as above, plus 2d (two back-to-back loads through an evicted alias) and 2e (evicted load then `ebreak`: value intact, then cause 3); 7 replays, at most 1 per pc; 0 marker leaks into `exception_o`; 0 timeouts; refusal record latched id `0x616` arm `0001`; R-35 fixture 7; lint PASS | `results/sim-replay-8f6a0af98.result-lines.txt` — written when the variant batch (tieoff / noclear / mgen / mto / noflush / flushproxy / noleakgate + sweep) finishes |
| synthesis, second fix | pre-registered: the loop set unchanged by identity; LUTLP-1 = 0; ORDER test 0/500; WNS within a few ns of −10.615; `lsu_i` LUT delta within +300..600 | `results/synth-8f6a0af98.result-lines.txt`, once run |

Values come from the CAPPRINT registers in the retirement trace and from the `R43 ...` trace lines;
`cva6.py`'s return code and its "SUCCESS" line carry no information here.

## Board acceptance — pre-registered before the reflash

One boot each, ordered by the `board-run` skill, `k800` first in every boot:

1. **R1 harness B3 and B4** (the two images that trapped on r42b3/r42b4) complete with their QEMU oracles.
2. **live16 / live128 / live512** return `544 / 4352 / 17408`, no trap.
3. **P1 cell 6 `-O2`** completes with its oracle: hash `112006 38bb59fd`, lookasides 25,010.
4. **The R-35 stale probe** (image `35fb3fec3196841b`) STILL traps cause 25 at `+0x4354`, last in its boot,
   and the refusal record reads LATCHED with its id — the record's positive control. A passing boot before
   it reads `10` (empty) — the negative control.
5. The ladder (every retval as `../R42-…`'s bootA) and P1 cell 5.

**Refuted if** any live arm traps 25 (read the record: the arm bit says whether it was DEAD, a timeout or a
genuine hit-dead), the R-35 probe reads data, or the record reads `11`/`00`/non-one-hot.

## Reproduce

In a capstone-ariane worktree on branch `r43-query-on-miss` at `8f6a0af98`:

```bash
# cva6-build-rv container (docs/ONBOARDING.md); on apollo pin it: --cpuset-cpus=0-7,32-39
python3 cva6.py --testlist=../tests/testlist_r43.yaml --test r43-evict-live \
  --iss_yaml cva6.yaml --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
  --isscomp_opts="+define+S12_MEM_DELAY=12+R43_TRACE" --sv_seed 1 --iss_timeout 7200 \
  --issrun_opts=+time_out=20000000
```

`+R43_PROBE_TIEOFF` on the define list is the bisect control: every miss denies, no probe — R-35's
deny-on-miss, with R-45's flush still in. Read results from the CAPPRINT registers in the retirement trace
and from the `R43 ...` lines in the `.log.iss`.
