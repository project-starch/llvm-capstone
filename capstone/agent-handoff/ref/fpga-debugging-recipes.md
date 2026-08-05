# Debugging large applications on the Capstone FPGA — recipes

**Scope.** A permanent playbook for "a big program fails somewhere on the CVA6+Capstone board and we don't know where." Everything here is grounded in what this project actually did, with file:line or log citations you can re-read. Where a number is measured, the measurement is named. Where something is unknown, it says UNRESOLVED.

**Companion documents.** `CLAUDE.md:106` ("Debugging a blocker: BATCH VARIANTS, and make every run RETURN") is the one-page policy; this is the procedure. `capstone/agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md` is the board-driver contract. `ref/ISSUES.md` is the defect register. Read this file before your first board session, not after.

---

## 0. The economics — why the method looks like this

Every rule below is a consequence of three measured facts.

**JTAG load is a flat, size-proportional cost.** OpenOCD prints its own rate on every boot:

```
board-O1.log     (13:31)  downloaded 17466376 bytes in 132.506927s (128.725 KiB/s)
board-clamp.log  (15:03)  downloaded 17466376 bytes in 131.401016s (129.809 KiB/s)
board-refix.log  (16:27)  downloaded 27952136 bytes in 211.713486s (128.934 KiB/s)
```

128.7–129.8 KiB/s, constant across a 60 % size change. The firmware on disk right now is **30,049,288 bytes** (`capstone/caplifive-system/sw/buildroot/build/build/opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin`) ⇒ ~227 s of JTAG per boot.

**Linux boot is linear in initramfs size.** The kernel unpacks a built-in CPIO, and `Run /init as init process` moves with it, from three clean same-day points:

| initramfs | `Run /init` |
|---|---|
| 35,115,008 B (`board-s14.log`) | 100.68 s |
| 41,271,296 B (`board-refix.log`) | 113.28 s |
| 45,888,512 B (`board-burst.log`) | 126.20 s |

Slope ≈ **2.37 s per MB**. This morning's lean image (10,490,880 B, `board-clamp.log`) booted to `/init` in ~45 s.

**Therefore one boot costs ~6 minutes before a single domain instruction executes**, and it is getting worse by itself: the overlay `capstone/caplifive-system/sw/buildroot/overlay/test-domains/` has grown to 26 `.dom` + 1 `.user` = 41.8 MB, so today's accumulation alone added ~95 s of JTAG and ~81 s of boot to *every* iteration.

The consequence that drives everything else: **a board session that returns one bit is a catastrophic use of the resource.** Six sessions were spent that way on 2026-07-31 and produced nothing usable — `run_sqlite_stages_fpga.py:4-8`, and the six were bisecting the wrong function the whole time. Maximise bits per boot; never optimise anything that weakens a freshness gate, because a session that measures the wrong binary returns *negative* bits.

---

## 1. The default loop

"Large app fails somewhere on the board, we don't know where."

### Step 0 — Reproduce under QEMU through the *identical* controller, first

If it also fails under QEMU, stop: it is not a silicon question and the board is the wrong instrument. The gate is `capstone/benchmarks/sqlite/run-sqlite-silicon.sh`, which runs the silicon-config domain through `run-domain-smoke.py` with the same five success markers the board run uses (`run_sqlite_baked_fpga.py:143-146` — "same criterion here so a silicon pass means the same thing a QEMU pass does").

"Identical controller" is load-bearing. `capstone/agent-handoff/ref/ISSUES.md:769` (I-3) records that diagnostic probes were *board-only* for weeks because the QEMU loader entered the domain through the plain call path while the board entered through an annotated region share — "the share IS the entry". Four fixes were attempted before the actual cause (a trailing `call_dom`) was found. Two board boots produced one data point between them because of it.

### Step 1 — Make the failure produce a marker instead of silence

A wedged domain emits nothing. The only thing a failed run tells you is "somewhere after the last marker." Before anything else, ensure the host side prints phase markers that can physically escape:

- `write(2)`, never `fprintf` — `capstone/benchmarks/sqlite/sqlite_host.c:9-29`. stderr **resets the core** on this board; fd 1 does not.
- every marker **≤ 16 bytes**, the 8250 TX FIFO depth, so one FIFO load carries a whole marker (`sqlite_host.c:13-24`). The observed failure mode was `"sqlite-host: cre"` — exactly 16 bytes — followed by the bootrom banner.
- values go on their own following line, so a lost value never costs a lost phase.

Current marker ladder: `A dom-ok, B mkregion1, C mkregion2, D mapped, E share1, F share2, G enter, H return, X fail` (`sqlite_host.c:27-29`).

### Step 2 — Write a **staged-return** variant set

Not more wedge probes. Each variant runs the first N steps and **returns a marker**:

```c
*res = 0x5A6E0000u | (STAGE << 8) | (run_sqlite_staged(STAGE) & 0xff);
```

`sqlite_capstone_domain.c:507-516`. `ss` = stage reached, `rr` = the library rc at that point; the host prints it as `SQ: obs=<decimal>` and the runner decodes it (`run_sqlite_stages_fpga.py:58-62`).

Rules for writing the set (`sqlite_capstone_domain.c:239-247`, `CLAUDE.md:131-135`):

- the staged logic lives in a **separate function** (`run_sqlite_staged`), never `#ifdef`s threaded through the production path — otherwise the bisection is about a build that does not matter;
- stages ascend and each is a real boundary of the thing under test (stage 7–10 split `sqlite3_initialize()` at *its own* internal boundaries: `sqlite3MutexInit`, `sqlite3MallocInit`, `sqlite3PcacheInitialize`, `sqlite3RegisterBuiltinFunctions` — `sqlite_capstone_domain.c:280-305`);
- include at least one variant that returns a **bitmap** rather than a boolean, so a partial failure is distinguishable from a total one. Stage 4 returns which of {first, middle, last} heap byte did not survive (`:259-270`); stage 14 returns a bitmap of `s[1..8]` being non-zero, so one 8-bit return says exactly how far good data extends (`:342-355`). Stage 14 is what converted "strlen returns 1" into "only byte 0 survived the copy."

### Step 3 — Build every variant, each to its own output directory and its own name

```bash
for N in 7 8 9 10; do
  OUT_DIR=/tmp/capstone/sqlite-stage$N \
  DOMAIN_EXTRA_DEFS=-DCAPSTONE_SQLITE_STAGE=$N \
    bash capstone/benchmarks/sqlite/build-sqlite-silicon.sh > /tmp/capstone/build-stage$N.log 2>&1 &
done; wait
```

`DOMAIN_EXTRA_DEFS` reaches `sqlite_capstone_domain.c` because it is `#include`d into the amalgamation TU (`build-sqlite-silicon.sh:155-160`). `OUT_DIR` is honoured at `build-sqlite-silicon.sh:33`.

These builds are independent, so run them in parallel — measured 2026-07-31: `sqlite-stage11/12/13/obj/amalgam.o` all landed within 91 ms of each other. **But** the compile is single-threaded per TU and the box has a documented parallelism cap; do not exceed it (`ninja -j90`, never `-j112`).

### Step 4 — QEMU-gate every variant before it is allowed near the board

Each variant must run to its expected marker under QEMU. A staged build that would have wedged QEMU too is a build you wasted a boot on. This is the same discipline the ladder work used: "Every build below was QEMU-validated through `run-ladder-perf-qemu.sh` — the *same* controller the board uses — before it was allowed on hardware" (`history/27-07-2026_04-33-58_RESULTS-delin-wedges-the-RTL-controlled-and-second-fault-isolated.md`).

### Step 5 — Stage them all into ONE firmware generation

```bash
CARVE_BUDGET=1000 bash capstone/benchmarks/sqlite/stage-sqlite-in-rootfs.sh   # canonical pair, gated
cp /tmp/capstone/sqlite-stage$N/sqlite_silicon.dom \
   capstone/caplifive-system/sw/buildroot/overlay/test-domains/sqlite_stage$N.dom   # each variant
sha256sum capstone/caplifive-system/sw/buildroot/overlay/test-domains/sqlite_stage*.dom \
   | tee /tmp/capstone/staged-shas.txt
```

Record the hashes. **They are not checked by anything today** — see §3, A5. Recording them is the minimum until the gate is extended.

Then rebuild the firmware. The relink consumes `build/images/Image`, and `Makefile:32-36` runs `$(A)` **before** the bare buildroot pass, so a single `A=opensbi-rebuild` links the *previous* generation's kernel:

```bash
cd capstone/caplifive-system/sw/buildroot
make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../capstone-c)"
make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../capstone-c)"   # yes, twice
```

`LINUX_PAYLOAD=1` is required (`stage-sqlite-in-rootfs.sh:110-112`: without it you get a ~2 MB image that boots to nothing). If `sbi_capstone.c` changed, `opensbi-rebuild` only **relinks**, so delete the stale `fw_*.o` first — `HOW-TO-LAUNCH-ON-FPGA.md:108-120`, issue C-11 (`ISSUES.md:1105`).

The double invocation is ~27 s of pure duplicated work (all 24 `/tmp/capstone/fwbuild-*.log` are byte-identical — single md5 across the set). It is cheap insurance and the freshness gate catches the wrong order anyway; **do not remove it without validating the reorder against the gate first**.

### Step 6 — Pre-flight (see §5). Do this before taking the lock.

### Step 7 — One boot, all variants, ascending, cheap first

```bash
FPGA_URL=<FPGA-CONSOLE-URL> \
FPGA_FW=capstone/caplifive-system/sw/buildroot/build/build/opensbi-custom/build/\
platform/fpga/ariane/firmware/fw_payload.bin \
SQLITE_STAGE_DOMS=/test-domains/sqlite_stage7.dom,/test-domains/sqlite_stage8.dom,\
/test-domains/sqlite_stage9.dom,/test-domains/sqlite_stage10.dom \
PROBE_SCOPED_OUT=/tmp/capstone/sqlite-stages789.txt \
  /tmp/capstone/fpga-venv/bin/python \
  capstone/tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py 2>&1 \
  | tee /tmp/capstone/board-stages789.log
```

Ordering is load-bearing (`run_sqlite_stages_fpga.py:36-37`, `:17-19`): a wedged domain takes the core with it, so everything after the first wedge is lost — **that is not a limitation to engineer around, the first variant that fails to return IS the bisection point.** Put anything you expect to hang last. `SQLITE_STAGE_TIMEOUT` defaults to 90 s per domain (`:43`), short on purpose: a staged build that will return does so immediately.

### Step 8 — Read the answer, then write the next stage set *before* the next boot

The runner prints `FIRST FAILURE: <dom> did not return. Everything below that stage works on silicon; the fault is inside that step.` (`:134-137`). Then split *that step* at its own internal boundaries and go back to Step 2.

This is what the ladder actually looked like on 2026-07-31, and each rung's brackets were authored *from the previous board result* — stages 0–3 → 4–6 → 7–10 → 11–13 → 14. That is why "just batch 12 variants" does not work: the brackets do not exist in advance. Batch as wide as the information justifies, no wider.

**The between-boots work is where the value is.** The 15:06→16:10 window contains five board sessions and four hypotheses killed *offline* with no board time at all (commit `06598df8c39d`).

---

## 2. Recipes

### R1 — Staged early-return builds, batched into one boot

**WHEN.** Any "fails somewhere in a large body of code" problem. This is the default; deviate only with a reason.

**HOW.** §1 Steps 2–8. Marker `0x5A6E_ssrr`. Runner `run_sqlite_stages_fpga.py`.

**WHY IT WORKS.** Two separate wins. (a) *Information*: a build that returns always yields a result, so the bisection converges instead of guessing (`sqlite_capstone_domain.c:509-513`). (b) *Cost*: ~6 minutes of boot amortised over N variants instead of paid N times (`run_sqlite_stages_fpga.py:12-15`).

**HOW IT FAILS.**
- **The variants are not freshness-gated.** `assert_firmware_embeds_current_initramfs` checks exactly two hardcoded files — `run_sqlite_baked_fpga.py:94` loops over `(LOCAL_HOST, LOCAL_DOM)`, defined at `:112-113`. No `sqlite_stageN.dom` is ever verified, in the firmware or on the board. `run_sqlite_stages_fpga.py` has no `sha256sum` step at all.
- **A missing variant is not loud.** `{HOST} {dom}; echo DN_$?` with a missing file gives `DN_127`, and `DN_\d` matches the `1`, so `run_command` returns normally, `wedged=False`, `obs=None`, and the first-bad test at `:120` never fires. The run then prints "Every domain in this set returned rc=0 … The failure is outside what these stages cover." That is a confident false pass from a session that tested nothing. **Fix before relying on batch width:** extend the gate to take the actual `DOMS` list, and treat `not found` / `DN_127` as a hard stop with its own verdict.
- **Out-of-order stages lie.** `sqlite-stages456.txt` ran 4, 6, 5 — harmless because all three returned, a wrong-answer path if one had not. The comment says "keep them ordered or the 'first failure' logic lies" (`:36`); nothing enforces it.
- **Size cannot tell variants apart.** `sqlite_stage11/12/13/14/14ctl/14fix.dom` are all exactly 1,538,952 bytes. `run_sqlite_baked_fpga.py:189` records the same trap: "a stale and a current domain were both 1623008 bytes on 2026-07-30."

### R2 — Convert a hang into a wrong answer (clamp / early return / bounded loop)

**WHEN.** The failure is a wedge, and observing the wedge has stopped producing new information. `CLAUDE.md:136-138` states this as a corollary: prefer a diagnostic that converts a hang into a wrong answer over one that only observes the hang.

**HOW.** Give the runaway a bound and let it return a wrong value:

```c
#ifdef BEEBS_STRLEN_CLAMP
  while (i < (bsize_t)(BEEBS_STRLEN_CLAMP) && s[i]) i++;
#else
  while (s[i]) i++;
#endif
```

`capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c:158-181`. Pick the bound **well above any legitimate value** (64 KiB, against a widest legitimate string capability of 256 KB) so a clamped return is unambiguously the pathological case and never a normal one — and so QEMU can never reach it, which the QEMU gate then checks for you.

**WHY IT WORKS.** A wedge hides *everything after it*: one bad string conceals the entire rest of the run. Clamped, the domain keeps going and answers the question the wedge cannot — is this ONE bad case with a working library behind it, or the first of many? On this project the clamp is what showed `strlen` was not even spinning, i.e. all six preceding sessions had been bisecting the wrong thing (`sqlite_capstone_domain.c:509-513`).

**HOW IT FAILS — and this one has already cost hours.** A diagnostic clamp that ships is indistinguishable from a hardware defect. `INTERP_BUILD_LIMIT=900` clamps the entry glue's carve loop; the table is still carved at full size, so slots 900..1058 are untagged memory, an `ldc` from one yields an untagged capability, and the first `stc` through it **stalls the pipeline with no trap** — pc frozen, `mcause=0`. That was "investigated as a hardware defect for hours. It was our own diagnostic knob" (`stage-sqlite-in-rootfs.sh:74-83`). The clamp is invisible in the descriptor (`count` still reads 1059), so the guard has to read the *entry code* of the artifact: `gp-carve-count.py:70-100` greps for the `li t3,<N>; bge t3,s4; mv s4,t3` sequence. Staging refuses a clamped canonical domain (`stage-sqlite-in-rootfs.sh:86-92`).

Two more failure modes, both live today:
- **Clamped variants under their own names are not checked at all** — `sqlite_lim1.dom`, `sqlite_lim512.dom`, `sqlite_nocapinit.dom` (Jul 30 21:39) are still in the overlay and still ride into every firmware built today.
- A clamp is *never correct*: a string longer than the clamp gets the wrong length. It answers a question; it is not a fix and must not be left in.

### R3 — A/B control builds: did MY change cause this?

**WHEN.** A symptom appears after your own change; or a one-instruction delta appears to flip a result. On this machine, do not skip it — a four-instruction change *outside the computation* flipped a passing rung to a deterministic wrong answer (`history/26-07-2026_17-43-17_controlled-ab-four-instructions-flip-a-passing-rung.md`).

**HOW — three levels, in increasing strength.**

1. *Twin build.* Same rung, same kernel, same compiler, same flags, differing only in `domain_main`. `beebs_prime` (+2 `csrr minstret`, +3 stores) → 1087631800 ❌; `beebs_prime_noins` → 582955588 ✅ = oracle. The control is a first-class rung (`-DLADDER_NO_MINSTRET`) so it passes the same QEMU gate and the same build path as its twin.
2. *Size- and position-matched control.* When the delta is one instruction, hold everything constant and change only the **encoding**:
   ```c
   /* #67c */ __asm__ volatile(".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(A));  /* delin  */
   /* #67e */ __asm__ volatile(".insn i 0x13, 0x0, x0,  0(x0)"      : "+r"(A));  /* addi 0 */
   ```
   Both 4 bytes, same position, identical `"+r"(A)` plumbing; the emitted functions differ in exactly one line. `#67c` HANGS, `#67e` RETURNS ⇒ layout, size, register allocation and surrounding code are eliminated (`history/27-07-2026_04-33-58_...md` §1).
3. *In-sweep known-good control.* Keep a rung that must pass in every batch. `beebs_bs` was included purely as a stability check, and its failure is the only reason a misconfigured sweep was caught instead of believed (`ISSUES.md:731-736`, I-1).

**WHY IT WORKS.** It converts "my change correlates with the symptom" into "these two artifacts differ by exactly X and disagree." Everything else — layout sensitivity, icache state, boot-to-boot variation — is held constant by construction rather than by argument.

**HOW IT FAILS.** If the twin is built by a different path, or one side is rebuilt and the other reused, you have measured your build system. Both sides must be built and QEMU-gated in the same command, and both hashes recorded. Also: a control that is *too* similar can mask the effect — stages 11/12/13 are near-identical binaries (`Segment size = 14db48`, `Loadable size = 1366856` identical across all three), which is fine for a bisection and useless as an icache-staleness control.

### R4 — Hash-gate every artifact before a board run

**WHEN.** Always. There is no run cheap enough to skip this.

**HOW — four gates, three of which already exist.**

1. **Firmware embeds the current initramfs.** `run_sqlite_baked_fpga.py:49-107`: locate the gzip member inside `fw_payload.bin`, **decompress it**, and search for the actual bytes of the binaries about to run. Three cheaper checks were tried and all fail: mtimes (fw/gz/Image land ~13 s apart in one `make`), `fw.find(Image[:4096])` (kernel header, identical across generations), `fw.find(rootfs.cpio.gz[:4096])` (gzip embeds an mtime, so it reports every firmware stale). Read `:64-72` before proposing a fourth.
2. **The booted image is the uploaded image.** `run_sqlite_baked_fpga.py:196-215` runs `sha256sum` **on the board** and compares to the local build, then refuses with `STALE BOOT — … Do NOT interpret any failure past this point as a domain or monitor bug.` `ls -l` sizes are explicitly not enough (`:187-189`).
3. **The image name is a content hash.** `_hash_name()` at `:33-45`: `fw_<sha256[:12]>.bin`. Identical firmware reuses the stored image; changed firmware gets a new name automatically. A timestamped name proves nothing about freshness.
4. **The artifact carries no diagnostic knob.** `gp-carve-count.py`, invoked by `stage-sqlite-in-rootfs.sh:86`.

**WHY IT WORKS.** Content-addressing is the only thing that survives this build system. `make build A=opensbi-rebuild` links the payload *before* buildroot regenerates the images (`Makefile:32-36`, `objects.mk:26` `FW_PAYLOAD_PATH=../../images/Image`), so the firmware embeds the previous generation by construction. On 2026-07-30 the freshly staged `sqlite_host.user` was in `rootfs.cpio` and absent from the firmware; the board answered `-sh: /test-domains/sqlite_host.user: not found`, exit 127, and the run "tested nothing while looking like a domain failure" (`run_sqlite_baked_fpga.py:52-58`).

**HOW IT FAILS — known holes, do not assume coverage you do not have.**
- Gate 1 covers **two files only** (`:94`, `:112-113`). Every staged variant is outside it.
- Gate 1 **warns and returns** rather than raising when no gzip member is found (`:82-84`), and **silently `continue`s** when a local file is missing (`:95-96`).
- Gate 2 is **conditional**: `checks` is empty when `SQLITE_HOST` is set and `SQLITE_DOM` is overridden without `SQLITE_LOCAL_DOM` (`:196-200`), which is the documented PROBE MODE (`:161`). Pass `SQLITE_LOCAL_DOM` on any multi-variant session or you lose the gate exactly when you have most variants in flight.
- `IMG_NAME` is **not** always a content hash: `FPGA_FW_NAME` overrides it (`:46`), and `run_ladder_perf_fpga.py:49` hardcodes a fixed name.
- **Nothing hashes the kernel or the monitor.** A monitor-only rebuild under a fixed name passes every gate above.

### R5 — Completion sentinels, never process polling

**WHEN.** Every board run, and every wait on one.

**HOW.** Drivers print, in order: `RUN_DONE` / `PROBE_DONE` (first statement in `finally`, so it survives a throwing teardown — `run_sqlite_baked_fpga.py:266-268`), then `BOARD_RELEASED`. Poll for those strings, with a bounded loop:

```bash
for i in $(seq 1 120); do grep -q BOARD_RELEASED /tmp/capstone/board-x.log && break; sleep 15; done
```

Drivers exit via `hard_exit()` — `os._exit` after flushing — not `sys.exit()`, because socketio's non-daemon thread outlives `disconnect()` often enough that `sys.exit` hangs the interpreter (`HOW-TO-LAUNCH-ON-FPGA.md:226-232`).

Teardown is time-boxed and ordered least-important-first: switches, power, unlock, disconnect (`safe_cleanup.py:1-22`). A step that will not complete is abandoned and logged, because releasing the board matters more than tidying it.

**WHY IT WORKS.** "Is the board still busy?" was answered wrongly, repeatedly, and each wrong answer cost either board time or an idle session (`HOW-TO-LAUNCH-ON-FPGA.md:211-214`).

**HOW IT FAILS.**
- `pgrep -f <pattern>` matches the polling loop's own command line and spins forever. **Six such loops ran here for up to 21 hours** (`HOW-TO-LAUNCH-ON-FPGA.md:222-225`).
- A run that printed both sentinels and then stayed alive emitting `user_count` events was reported as phantom board activity — the board was free (`:226-232`).
- A cleanup step blocking inside `finally` strands the lock and everything queued behind it: `probe_revnode.py` sat **16 minutes** in `set_switches(console, 0)` with all four of its readings already on disk; `probe_sqlite_wedge.py` held the lock **24 minutes** after its results were written (`safe_cleanup.py:10-16`).
- An ad-hoc console script that never calls `disconnect()` holds a session forever — one lived **49 minutes** and looked like another user on the board (`HOW-TO-LAUNCH-ON-FPGA.md:190-195`). Always `disconnect()`, or run it under `timeout`.

### R6 — Choose an offline discriminator over a board one

**WHEN.** Before every board session. Ask: is there a question I can answer by reading a file, that would change what I run?

**HOW — the offline instruments, in order of how often they have paid off.**

1. **Read the RTL.** `capstone/capstone-ariane/core/anvil_build/*.anvil` and `vendor/.../*.sv` are in-tree and readable. This has *answered* questions outright:
   - `MOVC` writes `cnull` to the SOURCE when `rd != rs1` and the source is not `CAP_TYPE_NONLIN` — `capstone_flu_unit.anvil:6-27`, verbatim in the file. That is the whole of C-14.
   - `CINCOFFSET` returns `rs1` unchanged via `create_result_pack(..., rs1, rd)` — `capstone_flu_unit.anvil:29-46`. That is why the linear-safe `strlen` indexes instead of walking.
   - The shadow-tag store→load race was **retracted from the RTL source**: the AXI adapter interlocks, a load needing a tag read enters `TAG_WAIT` and holds until every outstanding tag write takes its B-response (`wt_axi_adapter.sv:406-427`). A drafted board-owner question was withdrawn as answered in-tree (`current-next-step.md`, "Ruled out this session").
   - `LDC` is a plain load after its checks (`capstone_dyn_unit.anvil:296-352`); "a double `ldc` consumes a linear cap" is not available as an explanation on this platform.
2. **Read the emitted code.** `llvm-objdump -d --disassemble-symbols=<fn>`. The R-1 discriminator sharpened from "register-addressed" to "two capability registers derived from the same object, load through one, store through the other" purely by reading the disassembly, board-free (`history/27-07-2026_17-05-00_...md`, "REFINED by reading the emitted code").
3. **Static artifact checks.** `gp-carve-count.py` answers "does this domain fit the 1024-entry rev-node pool?" in seconds instead of a ~35-minute firmware rebuild plus a board session (`gp-carve-count.py:16-19`).
4. **Run the probe under QEMU.** Since I-3 was fixed, `capstone-diag.user` reads `res[3..47]`, so the R-1 diagnostic family develops off-board: "probe iteration drops from ~2.5 min of a shared physical resource to seconds of emulation" (`ISSUES.md:800-806`).
5. **Read the device tree.** I-2's three silent board sessions ended when the UART parameters were taken from the firmware's own FDT (`/soc/uart@10000000`, `reg-shift=2`): "**The FDT had the answer on disk the whole time**" (`ISSUES.md:763-767`).

**WHY IT WORKS.** Board time is the scarcest resource in the loop and the only one that is shared and human-visible. An offline answer costs minutes of a machine nobody is waiting for.

**HOW IT FAILS.** QEMU is permissive where the RTL enforces — QEMU executed the `-O0` `strlen` shape happily while the board froze on it, and QEMU executes every R-1 probe correctly. So an offline check can *refute* but rarely *confirm* a silicon claim. Use offline work to eliminate candidates and to choose which single question the board answers, never to declare a silicon result.

### R7 — Register predictions before the board speaks

**WHEN.** Any board session intended to test a hypothesis rather than measure a number.

**HOW.** Write the predicted outcome and what each result discriminates, in the issue entry, *before* the run. Worked example: four rungs registered 2026-07-27 with a table of prediction + what it discriminates, "written down *before* the board speaks so they are tests and not stories" (`ISSUES.md:66-79`). The scored tally was **2 hits, 3 misses, 1 partial**, and it is that tally which established R-1 is not a complete account of the board's behaviour.

**WHY IT WORKS.** It makes a session falsifiable and prevents retro-fitting a story to whatever came back. It is also the only cheap way to notice that your model is wide of the mark.

**HOW IT FAILS.** Only if the predictions are vague. "Something will break" discriminates nothing; "`beebs_cnt` PASSES, which tests R-1's same-object clause" discriminates a specific clause.

### R8 — Many domains in one boot via **distinct entry VAs**

**WHEN.** Running several `.dom` in one session.

**HOW.** `LADDER_DISTINCT_VA=1` on the build (assigns `0x10000`, `0x20000`, … 64 KiB apart) **and** `LADDER_ONE_BOOT=1` on the runner. Both opt-in (`ISSUES.md:517-522`).

**WHY IT WORKS.** The multi-domain hang is **address-keyed**, not count-keyed: `beebs_bs`@`0x10000` then `beebs_prime`@`0x20000` back to back, no power-cycle, both returned their oracles. Validated as *measurement-safe*, not merely correct, by a reversed-order control — spread 0.75 % / 0.03 %, `instret` byte-identical in both positions (`ISSUES.md:505-513`). A 13-rung sweep goes from ~13 boots (~35 min) to 1 (~5 min).

**HOW IT FAILS.**
- **Same-VA reuse still hangs.** The monitor still lacks the icache invalidate on domain switch (`ISSUES.md:529-531`). Note the staged SQLite variants all load at `Entry address = 10000` — and same-boot sequences of them *have* run (`sqlite-unalign.txt` shows `sqlite_stage14fix.dom` returning `DN_1` and then `sqlite_silicon.dom` running as `id=1` in the same boot), but two wedges at positions 3 and 4 also appear in the same log family (`sqlite_stage2`, `sqlite_stage10`). **UNRESOLVED whether those wedges are content or position.** Treat a wedge at position ≥ 2 as suspect until reproduced as the first domain of a clean boot.
- **A wedged rung poisons the rest of the sweep unless recovery is enabled.** On 2026-07-28 a hang made the runner keep "reusing" a dead boot, losing the **four** rungs after it, all of which had worked minutes earlier. A timed-out rung must clear the boot flag (`ISSUES.md:514-518`). Anyone re-implementing one-boot mode must include this.
- Do not make it a default: if the address-keying assumption ever fails, the symptom is a silent hang that looks like a result.

### R9 — Probe the wedge inside the run you already have

**WHEN.** A run just wedged and you want registers.

**HOW.** `probe_sqlite_wedge.py` drives its own boot and attaches gdb while the wedge is still live (`monitor halt`, **never** `reset` — reset zeroes the CSRs that make it readable). `run_sqlite_baked_fpga.py` cannot be used for this: it powers the board off in its `finally`, destroying the state (`probe_sqlite_wedge.py:19-21`).

**Better, and not yet implemented:** when `run_sqlite_baked_fpga.py`'s `run_command` at `:239-241` raises `ActionTimeout`, the console, the lock and the boot are all still live through `:249`, and the outer `finally` at `:263-271` releases them. Folding the probe block into an `except` there (behind an opt-in flag) removes an entire ~200 s boot cycle per wedge investigated. Three mandatory gates if you build it: **re-raise** the original exception (`out` is already bound to the `ls -l` output at `:218`, so falling through would grep that text for markers and print a plausible `SQLITE ON SILICON: FAIL` instead of naming the wedge); **`gdb_stop()` in a `finally`** (a leaked running session makes the next `cold_boot`'s `gdb_start` burn a 60 s timeout, and `release_board` does not stop gdb); and keep it **opt-in**, since not every idle timeout is a wedge (`run_sqlite_baked_fpga.py:129-138` documents a live run aborted at 75 s idle mid-CREATE/INSERT).

**WHY IT WORKS.** The reboot is 100 % of the cost of a standalone probe and 0 % of its information.

**HOW IT FAILS.** `gdb_start` on a domain mid-Capstone-domain-switch can desync the session ("packet queue is empty, aborting" — `run_rtl_smoke.py:207-209`). Acceptable *after* a failed run — the worst case is a lost diagnostic, not a wrong number — and unacceptable before or during a measurement.

---

## 3. Anti-patterns, with the cost each one caused here

**A1 — One hypothesis per board session.** Six sessions narrowing a wedge inside `strlen`; each bought a single bit; the clamp then showed `strlen` was not even spinning, so all six had been bisecting the wrong thing. The answer came from one session that ran four variants. `run_sqlite_stages_fpga.py:4-8`, `CLAUDE.md:107-110`.

**A2 — Observing a wedge instead of forcing a return.** A wedged domain emits nothing, so every failed run says only "somewhere after `SQ: G/enter`" — and possibly about the wrong function. `sqlite_capstone_domain.c:508-513`.

**A3 — Believing a pc sample.** Two retracted conclusions from the same mistake.
- Commit `e03a3124` claimed the wedge was cleared because pc advanced across three `stepi` (`0x14cc74 → 78 → 7c`). Retracted: three back-to-back single-steps prove the debug module can force the core past an instruction; they say nothing about free-running execution. `probe_sqlite_progress.py` sampled five times over ~100 s with the core resumed in between: **same pc, `a0` unchanged at 31,342,951** — still wedged (`history/31-07-2026_14-00-00_...md`, "UPDATE, same day").
- The mirror error is equally documented: a tight loop resumed and re-halted can be caught at the *same* pc every time by a deterministic debug module, so an unchanging pc across `resume`/`halt` does **not** prove an instruction is stuck (`probe_sqlite_wedge.py:104-109`).

**A4 — Passing a knob as a command prefix.** `run-sqlite-silicon.sh:19` and `stage-sqlite-in-rootfs.sh:64` rebuild the domain **unconditionally**, so a flag set only as a prefix on an earlier standalone build is discarded on the rebuild. That is how a run tested the UNCLAMPED domain while the log said `limit=900` (`run_sqlite_baked_fpga.py:190-194`), and how `SQLITE_NO_TRIM` was silently restored (`build-sqlite-silicon.sh:122-125`). **EXPORT it, and check the artifact hash CHANGED before believing any negative result** (`current-next-step.md`, "Build traps that are still live").

**A5 — Running a stale or unverified artifact.** The single most expensive class here.
- 2026-07-25: the ladder-perf runner reused existing `<rung>.dom` files and reported **4 bogus "silicon miscompiles"** that were an already-fixed compiler bug (`HOW-TO-LAUNCH-ON-FPGA.md:98-103`).
- 2026-07-27: a sweep silently rebuilt at `-O0`, producing **five rungs reported as silicon failures**, a false conclusion that "R-1's same-object clause was refuted" that would have gone to the board owner as a correction to the bug report, and a nearly-published §5 claim that an ordinary rebuild flips a passing rung. All three withdrawn. Caught only by the in-sweep control rung (`ISSUES.md:719-743`).
- 2026-07-29: the FPGA image carried a domain a **day older** than the build, predating two fixes, because only the QEMU overlay had been staged (`stage-sqlite-in-rootfs.sh:27-33`).
- 2026-07-30: the firmware embedded the previous generation; exit 127 "looks like a domain failure" (`run_sqlite_baked_fpga.py:52-58`).
- **Live today:** three diagnostic clamped builds from Jul 30 (`sqlite_lim1/lim512/nocapinit.dom`) are still in the overlay and still ride into every firmware; no gate covers them.

**A6 — `pgrep -f` polling.** Six loops, up to 21 hours (§R5).

**A7 — Abandoning a locked board.** 24 min, 16 min, and a 49-minute orphaned console script (§R5).

**A8 — Grepping the accumulated console log.** The console replays its history ring on connect — ~10 previous boots, ~524 KB in a 618 KB log file — so a naive `grep -c` over a board log counts other sessions' output. Multiple reviewers of this workflow independently produced wrong per-boot counts this way. Use the **run-scoped** capture: `console.uart_mark()` / `uart_since(mark)`, written to `SQLITE_SCOPED_OUT` / `PROBE_SCOPED_OUT`, and read *that* (`run_sqlite_baked_fpga.py:234-249` — "THIS RUN ONLY; do not grep the accumulated log"). A related trap: printks are split across `[fpga] [uart] '...'` chunks, so a regex anchored to one line silently misses boots.

**A9 — Letting the overlay grow.** Measured today: boot to `/init` went 44.8 s → 126.2 s, JTAG 132.5 s → ~227 s, purely from accumulated `.dom` files. That is ~176 s given away on **every** boot, forever, and it compounds. **But do not fix it by deleting first and gating second** — a pruned-away variant produces `DN_127`, which the stages runner reports as a *pass* (§R1). Order: extend the gate to cover every dom under test, then prune. The three provably dead Jul-30 doms (9.5 MB) can go today with no loss of A/B fallbacks.

**A10 — Sizing a UART transfer from the raw file, and using UART at all for large binaries.** Sizing SQLite from its raw 2.27 MB gave "≥ 63 min" and led to ruling UART out; the real figure was ~15 min. But ~15 min was also wrong in the other direction: at 703 K chars a dropped character is near-certain and `_put_once` **truncates and restarts the whole file**, then falls back to burst=1 (~4 hours). "UART's limit at this scale is **reliability, not throughput**" (`HOW-TO-LAUNCH-ON-FPGA.md:160-175`). Bake large artifacts into the initramfs and let them ride JTAG. Also: if a transfer suddenly looks slow, check for `burst=1` on the **first** attempt — that is the documented regression signature (`:47-60`).

**A11 — Reading normal behaviour as a failure.** Two documented false alarms that each cost session time: the board powering off at the end of a run is the runner's `finally`, not a crash; and a reboot banner right after a rung's marker is usually the *next* rung's power-cycle — look for a `power-cycle + reload firmware` line before the banner (`HOW-TO-LAUNCH-ON-FPGA.md:177-188`).

---

## 4. Instrumentation that works on this platform vs instrumentation that lies

### Works

| Instrument | Why it survives | Citation |
|---|---|---|
| `write(2)` markers ≤ 16 bytes | One 8250 TX FIFO load carries a whole marker; stderr resets the core, fd 1 does not | `sqlite_host.c:9-33` |
| Staged return marker `0x5A6E_ssrr` | Value travels back through the shared region, not the console | `sqlite_capstone_domain.c:507-516` |
| `mcycle` / `minstret` CSR reads | The measurement vehicle for all published silicon numbers | `ISSUES.md:745-760` |
| On-board `sha256sum` of the artifact | Proves the *running* system is the image you uploaded | `run_sqlite_baked_fpga.py:196-215` |
| Decompress-the-firmware freshness gate | Invariant to gzip mtime and to build generation | `run_sqlite_baked_fpga.py:49-107` |
| `gp-carve-count.py` | Reads the entry code, not the descriptor — catches a clamp the descriptor hides | `gp-carve-count.py:70-100` |
| `csdebugprint` capability-bounds print | Prints `Cap(type, perms, cursor, base, end)` directly — **QEMU only** | `beebs_freestanding_string.c:139-151` |
| The device tree on disk | Answered three sessions' worth of UART guessing | `ISSUES.md:763-767` |

### Lies, or cannot be used

**Debug-register reads can return AXI error-slave junk.** `0xCA11AB1EBADCAB1E` is `RespData` of the error slave — `capstone/capstone-ariane/vendor/pulp-platform/axi/src/axi_err_slv.sv:25`:

```systemverilog
parameter logic [RespWidth-1:0] RespData = 64'hCA11AB1EBADCAB1E, // Hexvalue for data return value
```

It means the read went to an **unmapped address** and the value is junk, not data. It has appeared twice; in one dump `$a1 = 0xca11ab1ebadcab1e` **and** `$mstatus = 0xca00000000` in the same read, alongside an `a0 = 0x0` that a mechanism story was being built on (`current-next-step.md`, "Caveat, do not skip"). **Treat any register whose value carries the `0xca11ab1e` / `0xbadcab1e` signature as UNREAD, not as zero and not as data.** `probe_sqlite_progress.py:66` flags them automatically (`BAD = (0xca11ab1ebadcab1e, 0xbadcab1e)`); `read_reg` returns `None` for them (`:79-90`). Any hand-driven gdb session must apply the same filter by eye.

**A pc sampled under `stepi` says nothing about free-running execution.** `stepi` forces one instruction to retire through the debug module; the core may not run at all when resumed. Cost: one wrong conclusion, published in a commit message and retracted the same day. **And the converse:** a pc that is *identical* across `monitor resume` / `monitor halt` does not prove an instruction is stuck, because a deterministic debug module can catch a tight loop at the same pc every time. Cost: a second wrong conclusion. `probe_sqlite_wedge.py:104-109`; `history/31-07-2026_14-00-00_...md` "UPDATE, same day".

> **The only sound liveness test on this board: samples separated by wall-clock, with the core resumed in between, plus a monotone counter register.** `probe_sqlite_progress.py` does exactly this — 5 samples, 20 s apart, reading pc *and* `a0` (strlen's index). Its three verdicts are the template: pc leaves the loop ⇒ progressing; pc stays and the counter climbs ⇒ spinning on one input (a *data* bug); pc stays and the counter does not climb ⇒ genuinely stuck (`probe_sqlite_progress.py:3-24`).

**`C_PRINT` (`csrw 0x800`) goes to the RTL trace, not the UART.** Do not use it as a UART probe (`HOW-TO-LAUNCH-ON-FPGA.md`, "Non-negotiables"). This is not cosmetic: `capstone_error` is `C_PRINT(...)` + `while(1)`, so **all five** monitor silent-spin sites are indistinguishable from a hang on the board — `handle_interrupt` default (`sbi_capstone.c:898-900`), `handle_exception` default (`:973-977`), illegal-instruction-not-`time` (`:959-963`), `swap_cpmp` (`:917-923`), and two in `split_out_cap` (`:236, :246`). Issue I-5 (`ISSUES.md`, "every monitor error is invisible") records a zero-board-cost fix — give `capstone_error` a real UART putchar via `split_out_cap(0x10000000, 0x100, 0)`, the same mechanism the monitor already uses for `mtime` — and calls it "the highest-leverage change available for board debugging." It is still open.

**`csdebugprint` (funct7 0x43 on opcode 0x5b) is not decoded by the FPGA.** A board build must never set `BEEBS_STRING_DEBUG_BOUNDS` (`beebs_freestanding_string.c:139-151`). Related: illegal/meaningless capability ops **wedge rather than trap** on this board (R-5, `ISSUES.md:539`), so an undecoded instruction does not give you an error — it gives you a dead board and a mystery.

**`ls -l` size comparison.** Two different domains were both 1,623,008 bytes on 2026-07-30 (`run_sqlite_baked_fpga.py:187-189`); today nine staged variants are all 1,538,952 bytes. Size is a warning at best; hash or nothing.

**mtimes.** `fw`, `.gz` and `Image` land ~13 s apart inside a single `make`, which reads as "same build" (`run_sqlite_baked_fpga.py:65-66`).

**Silence.** "Silence means wedged" was true when the domain froze at a pinned pc and is no longer true: SQLite between `SQ: G/enter` and its first row is legitimately quiet while it opens the database and runs CREATE/INSERT. A real run was aborted at the 75 s idle limit on exactly that stretch and was nearly read as another wedge (`run_sqlite_baked_fpga.py:129-138`). Raise `SQLITE_RUN_IDLE` only after the progress probe shows the core is live.

**The UART itself is lossy.** The RX FIFO overruns on a bulk write and silently drops characters — `borrow_cost` has arrived as `row_cost` (`fpga_console.py:759-762`). Markers must tolerate one dropped character: `config.py:243-245` uses `r"measurement.?complete"`, with the `.?` absorbing the missing character at the fragile hyphen.

**The rev-node pool's overflow flag.** `overflow_flag` reaches only a debug LED (`cva6.sv:1185`) — nothing traps, nothing prints. Allocation #1025 wraps to node id 0 and reuses live ids; the result is silent corruption, and since every `stc` blocks on a revocation-node query with no timeout (`capstone_dyn_unit.anvil:395-404`), the next capability store hangs with no trap. R-12, `ISSUES.md:1188`. Answer this **offline** with `gp-carve-count.py`, never on the board.

---

## 5. Pre-flight checklist — run every item before taking the lock

Board time starts when you take the lock. Everything below is free.

1. **Is the board free?** Board sessions are serialized across lanes and never parallel. Check that the previous session printed `BOARD_RELEASED`.
2. **QEMU gate green** for every artifact in the batch, through the same controller the board uses.
3. **`FPGA_FW` set to the right file.** `.../opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin` (~17–30 MB), **not** `build/images/fw_jump.bin` (~569 KB, the QEMU monitor, will not boot the board). It has no default on purpose; missing it throws at *import*, which means a stale `sqlite-run-scoped.txt` from an earlier session is still on disk and reads exactly like a fresh result (`HOW-TO-LAUNCH-ON-FPGA.md:199-210`). **Always confirm the driver actually ran before reading its output file.**
4. **`FPGA_URL` set for this run only** — it is a credential; never commit, never echo into a capture, never persist to disk.
5. **Two firmware relinks done**, in that order, after staging (§1 Step 5).
6. **If `sbi_capstone.c` changed:** the stale `fw_*.o` deleted, and the two `readelf` assertions pass (`HOW-TO-LAUNCH-ON-FPGA.md:108-120`).
7. **Freshness gate passes locally.** `assert_firmware_embeds_current_initramfs` runs before the lock in every driver; make sure you saw its `firmware carries the current binaries (initramfs N bytes, verified by decompressed content)` line.
8. **`gp-carve-count.py` clean** — no `INTERP_BUILD_LIMIT` baked in, `max index < count`.
9. **Every variant under test exists in the overlay AND its sha256 is recorded.** No gate does this for you today. Do it by hand:
   `sha256sum <every dom in SQLITE_STAGE_DOMS>` and compare against the build outputs.
10. **`SQLITE_LOCAL_DOM` set** if you override `SQLITE_DOM`, or you silently lose the on-board stale-boot gate (`run_sqlite_baked_fpga.py:200-203`).
11. **Domain order ascending**, cheap/safe first, the one you expect to hang last.
12. **Per-domain timeout appropriate.** 90 s for staged returns; raise the *idle* timeout only if the progress probe has shown the core live.
13. **Scoped output path set** (`PROBE_SCOPED_OUT` / `SQLITE_SCOPED_OUT`) and distinct from the last run's.
14. **A known-good control in the batch** if this is a sweep rather than a bisection (I-1's lesson).
15. **Predictions written down** (§R7).
16. **Bitstream expectation**: the driver hard-stops unless the resident bitstream matches
    `FPGA_BITSTREAM`, which defaults to **`caplifive_fixed_forward.bit`** since the
    2026-08-04 reflash (it replaced `working-caplifive-captype-fixed.bit`). Re-flashing a shared board is never automatic — ask first.
17. **Wait plan**: a bounded loop on `RUN_DONE` / `BOARD_RELEASED`. Not `pgrep`.

---

## 6. Telling "this is an RTL bug" from "this is our bug"

The default assumption is **ours**. On this project, attribution has been revised more often than any other kind of claim — C-14 went "the RTL is buggy" → "the spec mandates it, the RTL is conforming, QEMU deviates" → "the spec is under-specified; the weight of evidence favours scalars being exempt, so the RTL's `MOVC` is probably an oversight — **but this must be put to the board owner as a QUESTION, not an accusation**" in a single day (`ISSUES.md:1222-1240`). R-2 was filed as an RTL defect and later re-explained as C-13, i.e. ours (`ISSUES.md:75`).

### The discriminators, cheapest first

**D1 — QEMU through the identical controller.** Necessary, nowhere near sufficient. "QEMU passes, the board fails" is the *starting* condition of every entry in the RTL section and also of most entries in the compiler section. It does not decide anything by itself.

**D2 — Read the RTL source for that exact instruction.** In-tree, free, and decisive more often than expected (§R6.1). Three candidate mechanisms were killed this way in one session; one drafted board-owner question was withdrawn as already answered.

**D3 — Compare RTL and QEMU semantics side by side.** If they differ, you have found a *divergence*, and the next question is which one the spec supports — not which one is wrong. If they agree and the board still misbehaves, your model of the instruction is wrong, not the hardware.

**D4 — Size-, position- and encoding-matched control on the board.** The strongest single board experiment available. `#67c` (`delin`, `.insn r 0x5b,0x1,0x3`) HANGS; `#67e` (`addi x0,x0,0`, `.insn i 0x13,0x0`) at the same position with identical `"+r"(A)` plumbing RETURNS 9. The emitted functions differ in exactly one line. Layout, size, register allocation and surrounding code are eliminated by construction (`history/27-07-2026_04-33-58_...md` §1). **Do not skip this on a one-instruction delta** — this machine is documented to flip a passing rung on a four-instruction change outside the computation.

**D5 — Factor the ingredients until neither alone fails.** The R-1 isolation is the template: register index alone passes (v5 A), an extra store alone passes (v5 B), together they fail (P4). Ordering irrelevant (D), index arithmetic irrelevant (E), reproduces on a fresh boot (C). Nine probes, one conclusion (`history/27-07-2026_17-05-00_...md`).

**D6 — Try to mitigate it in software; failure to mitigate is itself diagnostic.** Seven mitigations for R-1 all failed — fences before the load, fences after every store, register hoisting, making the other store register-indexed, 64 B cache-line separation, constant-offset pointer walk, both accesses through pointers. "**Fences not helping is diagnostic**: this is not a memory-ordering problem, it is address disambiguation in the load path." That also retired the `fence.i` line of enquiry for that issue.

**D7 — Determinism across boots.** A wrong answer that reproduces bit-identically across two sessions with a full power-cycle and firmware reload between them (1087631800, cycles 47,954 / 47,952) is a different animal from a flaky one. Non-determinism points at state, ordering or your own harness, not at combinational logic.

**D8 — A cross-object / known-good control in the same batch.** `beebs_cnt` keeps stores outstanding to two *different* globals through two capability registers; it was predicted to pass and passed, which is what makes R-1's "same object" clause **tested rather than inferred** (`ISSUES.md:41-46`).

### It is probably OURS if any of these hold

- The artifact was not hash-verified end to end (A5).
- A build knob was passed as a command prefix rather than exported (A4).
- The symptom appeared with a compiler or ABI change and no size-matched control was run (D4).
- The failing construct is one the compiler emits and the ISA does not obviously support — e.g. `movc` used as a *copy* for a still-live source, when `MOVC` is defined as a MOVE (`capstone_flu_unit.anvil:6-27`).
- A capability arrives with bounds nothing in the program can justify. `strlen` scanned **31,342,951** bytes in-bounds without faulting, against a 1.37 MB domain image ⇒ the capability has bounds ≥ 31 MB, ~23× the whole domain. That is not an RTL fault; it is a pointer being formed by `auipc`/`lla` against whatever capability is in the base register instead of a bounded cap-table entry (`history/31-07-2026_14-00-00_...md`, "What the run DID establish").
- The failure disappears when a *diagnostic* is removed (the `INTERP_BUILD_LIMIT` case).

### It is probably the RTL if all of these hold

- Both implementations were read and they **differ**, or the RTL source contains the mechanism verbatim.
- QEMU is correct through the identical controller for **every** probe in the family.
- An encoding-only, size- and position-matched control passes where the suspect instruction fails (D4).
- Neither ingredient fails alone; only the combination does (D5).
- It reproduces on a fresh boot, and a reversed-order or cross-object control behaves as predicted (D7, D8).
- No software mitigation works, and the *pattern* of which mitigations fail is consistent with the proposed mechanism (D6).

### Even then

Write it as a question to the board owner, with the minimal reproducer and the control that eliminates layout. R-1's own entry still says "Confidence it is hardware: **high, not certain**. Residual doubt is whether our non-standard gp-captable ABI provokes it" (`ISSUES.md:29-31`). And keep the standing position that there are **≥ 2 independent faults**: R-1 speaks to memory-shape failures and does not explain the hangs (`beebs_janne`'s failing loop nest contains no memory operations at all — R-6, `ISSUES.md:440`). Do not let one characterised fault absorb every unexplained symptom.

---

## Appendix — where things are

| What | Path |
|---|---|
| Policy (batch variants, make every run return) | `CLAUDE.md:106-138` |
| Board-driver contract, traps, gotchas | `capstone/agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md` |
| Defect register (R-*, C-*, I-*) | `capstone/agent-handoff/ref/ISSUES.md` |
| Current position + next step | `capstone/agent-handoff/state/current-next-step.md` |
| Drivers | `capstone/tests/rtl-smoke/fpga_driver/` |
| — staged batch runner | `run_sqlite_stages_fpga.py` |
| — baked single run + all freshness gates | `run_sqlite_baked_fpga.py` |
| — wedge register dump | `probe_sqlite_wedge.py` |
| — progressing-vs-spinning, wall-clock sampling | `probe_sqlite_progress.py` |
| — bounded teardown, sentinels, `hard_exit` | `safe_cleanup.py` |
| Console protocol | `capstone/tests/rtl-smoke/socketio-api.md`, `fpga_driver/PROTOCOL.md` |
| Staged-return domain source | `capstone/benchmarks/sqlite/sqlite_capstone_domain.c:238-386, 505-517` |
| FIFO-safe host markers | `capstone/benchmarks/sqlite/sqlite_host.c:9-33` |
| Clamp / linear-safe string primitives | `capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c:112-192` |
| Build + stage + QEMU gate | `capstone/benchmarks/sqlite/{build,stage,run}-sqlite-*.sh` |
| Static artifact guard | `capstone/benchmarks/sqlite/gp-carve-count.py` |
| RTL sources (read these before asking) | `capstone/capstone-ariane/core/anvil_build/*.anvil`, `capstone/capstone-ariane/vendor/pulp-platform/axi/src/*.sv` |

**Two things this playbook does not yet have, and the next person should add:**
1. A freshness gate that covers **every** artifact under test, not the two canonical files — extend `assert_firmware_embeds_current_initramfs` to take the runner's dom list and search each inside the decompressed cpio it already extracts. Until that lands, batch width is limited by how many unverified binaries you are willing to have in the image.
2. `DN_127` / `not found` treated as a **hard stop with its own verdict** in every runner. Today it reads as a pass in one and as a domain failure in another; both are wrong.