# Minimising the FPGA debug cycle

Read-only investigation, 2026-07-31. No file was edited, no build run, the board was not touched.
All paths below are relative to `<REPO>/` unless absolute.
Every number is tagged **M** (measured — reproducible from a quoted file, log or mtime) or **I** (inferred — derived from a measured rate/fit, or bounded by accounting).

---

## 1. Where the time actually goes

Baseline = the state on disk right now, which is what the *next* iteration will cost:
`sw/buildroot/build/build/opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin` = **30,049,288 B** (M, `stat`, 16:30:09), `sw/buildroot/build/images/rootfs.cpio` = **45,888,512 B** (M, 16:30:16), overlay `sw/buildroot/overlay/test-domains/` = 28 files / **41,824,712 B** (M, `du -sb`).

| # | Step | s | M/I | Evidence |
|---|------|---:|:---:|---|
| **HOST (board free)** ||||
| H1 | Compile the silicon TU (9.5 MB amalgamation, `-O0`, 1 TU) | 42 | M | `/tmp/capstone/sqlite-silicon/obj/amalgam.c` 16:00:33.463 → `amalgam.o` 16:01:12.614 → `sqlite_silicon.dom` 16:01:15.514 |
| H1b | *N* extra stage variants | ~0 | M | stage11/12/13 `amalgam.o` land 15:38:12.866 / .866 / .96 — 91 ms apart, already parallel |
| H2 | Stage into the two overlays (`SKIP_BUILD=1` path) | 3.9 | M | `build-s14fix.log` 16:02:04.098 → `stage-unalign.log` 16:02:07.972 |
| H3 | OpenSBI relink (pass 1 — **links the PREVIOUS Image**) | 5.6 | M | `fwbuild-burst-1.log` closes 16:30:03.77 → `fw_payload.bin` written 16:30:09.39 |
| H4 | Bare buildroot pass (target/ → `rootfs.cpio` → `.gz` → `Image`) | 30.4 | M | 16:30:09.39 → `images/Image` 16:30:39.83 |
| H5 | **Duplicate** relink + bare pass (pass 2, the ordering workaround) | 26–37 | M | 8 log pairs today: 25.8 / 27.3 / 30.3 / 31.5 / 32.2 / 32.7 / 34.9 / 36.5 s; median 31.9 |
| | *host subtotal* | **~108** | | |
| **BOARD (locked, human-visible idle)** ||||
| B1 | `lock()` + resident-bitstream check | ~5 | I | `run_sqlite_baked_fpga.py:167-174`; no timestamps in log |
| B2 | HTTPS POST of the 30.0 MB image | 20–40 | I | never instrumented (`run_sqlite_baked_fpga.py:38` says "~30-90 s", unsourced). Bounded by session accounting: board-stages456 occupied ≤334.8 s of which 285.4 s is accounted M ⇒ upload of 19.56 MB ≤ ~40 s |
| B3 | Power cycle (off 8 + settle 15) | 23 | M | `run_rtl_smoke.py:62,65` constants, executed unconditionally in `cold_boot` |
| B4 | `gdb_start` + `monitor reset halt` + 4 s sleep | ~10 | I | `run_ladder_perf_fpga.py:155-160` |
| B5 | **JTAG `monitor load_image`** (30.0 MB @ 128.9 KiB/s) | **227.7** | I | rate is M: 13 loads today, 127.693–130.110 KiB/s across 15.37 / 17.47 / 27.95 MB. Directly M at 211.71 s for 27,952,136 B (`board-refix.log:50`) |
| B6 | OpenSBI → kernel entry | ~2 | I | no timestamp before `[ 0.000000]` |
| B7 | Kernel init to `clk: Disabling unused clocks` | 22.3 | M | 22.254–22.319 across **9** live boots spanning 10.49→41.27 MB initramfs — invariant |
| B8 | **initramfs unpack** (45.89 MB) | ~101 | I | fit is M: `Freeing unused kernel` vs initramfs = 44.79 s @10.49 MB … 113.24 s @41.27 MB ⇒ **2.224 s/MB**, 9 points, residuals <0.9 s |
| B9 | Userspace init → login prompt | 36.6 | M | `Run /init` → last printk: 36.69 / 36.65 / 36.62 / 36.52 / 36.35 / 36.63 / 36.67 (one outlier 38.51) |
| B10 | printk clamp, `insmod`, `sha256sum` gates, `ls -l` | 10–15 | I | `run_sqlite_baked_fpga.py:196-225`, 4–5 UART round-trips |
| B11 | Domain run(s) — returning | 5–20 ea | I | no `run took` line exists in any board log today (M: `grep -h "run took" /tmp/capstone/board-*.log` → empty) |
| B11′ | Domain run — **wedged** (the common case now) | 75 | M | `RUN_IDLE` idle timeout; every recent baked session ends `ActionTimeout: UART idle 75s … 'SQ: G/enter'` |
| B12 | Power off + unlock | ~10 | I | `safe_cleanup.py:75-92` |
| | *board-locked subtotal* | **~500** | | |
| | **ITERATION TOTAL (machine time)** | **~610 s** | | matches M end-to-end deltas 496.7 / 620.9 / 655.8 / 728.5 / 808.2 s (board-log → board-log), remainder = analysis/authoring |

Two facts to keep in view:

* **B5+B7+B8+B9 = 388 s (63%) is JTAG and boot.** Host build is 18%. Upload is at most 7%.
* **B5 and B8 are both linear in artifact size, and both artifacts have grown 1.7×–4.4× today** (17.47→30.05 MB firmware, 10.49→45.89 MB cpio) purely from overlay accumulation. That growth alone has added ~96 s of JTAG (I) + ~79 s of unpack (I) to *every* session since 13:31.

---

## 2. The optimised pipeline

### 2a. DOMAIN-ONLY-CHANGED path (the common case — new/edited `.dom`, no kernel, no OpenSBI, no monitor)

```
1. build the .dom(s)                                        42 s  M   (N variants in parallel, ~0 marginal)
2. stage into both overlays                                  4 s  M
3. ONE bare buildroot pass  (make build LINUX_PAYLOAD=1)    30 s  M   regenerates target/, cpio, .gz, Image
4. ONE opensbi relink       (make -C buildroot ... opensbi-rebuild)
                                                             6 s  M   now links the FINAL Image by construction
5. assert_firmware_embeds_current_initramfs (host, board-free)  <1 s
--- lock ---
6. bitstream check + upload                              5+20 s  I
7. power cycle + gdb + reset halt                           33 s  M/I
8. JTAG load_image (17.5 MB pruned)                        131 s  M   (227.7 s unpruned)
9. boot: 22.3 + unpack 22.5 + init 36.6                     81 s  M   (160 s unpruned)
10. sha256 gates + N domain runs                         25-95 s  I/M
11. power off + unlock                                      10 s  I
```

**New total ≈ 82 s host + ~330 s board ≈ 410 s** (I, composed of M parts), against ~610 s today.
Of the 200 s saved: **~32 s** from the relink ordering fix (safe, ship now) and **~175 s** from pruning the overlay (**gated** — see G1/G2 below; do not prune before they land).

Steps 3 and 4 are irreducible for this path: the initramfs is `CONFIG_INITRAMFS_SOURCE=…/rootfs.cpio` built *into* the kernel, so changing one `.dom` byte forces a full `Image` regeneration and a firmware relink. There is no domain-only shortcut, and the one that was proposed (JTAG-poking the `.dom` into DRAM) is rejected — §4.

### 2b. FULL-REBUILD path (OpenSBI / monitor `sbi_capstone.c` changed)

Identical, plus: **`opensbi-rebuild` only relinks, it never recompiles** (`ISSUES.md:1110`; `history/28-07-2026_16-10-00_monitor-regen-SOLVED-stale-fdt-object.md:13`). The documented `rm` of the stale `fw_*.o` objects (`HOW-TO-LAUNCH-ON-FPGA.md:113-120`) and the forced `.c.S` regeneration stay mandatory. The ordering fix does not touch this and does not excuse it. Budget +2–4 min.

### 2c. WEDGE path (currently ~every iteration)

Do **not** start a second board session to attach gdb. `run_sqlite_baked_fpga.py:239-249` is already inside the live console, live lock, live boot when `run_command` raises — see R2. Saves ~200 s per wedge investigated.

---

## 3. Ranked recipes — ordered by saving ÷ effort

### R1 — Stop paying for the double relink. **SAFE. Trivial. ~32 s/iteration (M).**

`sw/buildroot/Makefile:32-36` runs `make $(A)` **first** and the bare pass **second**, while `components/opensbi/platform/fpga/ariane/objects.mk:26` sets `FW_PAYLOAD_PATH=../../images/Image`. So the relink consumes the *previous* generation's Image and the recipe is invoked twice to converge. Proof on disk right now: `fw_payload.bin` 16:30:09.39 is **30.4 s older** than the `images/Image` it claims to embed (M).

Replace the two `make build … A=opensbi-rebuild` invocations with **one bare pass, then one relink**, shipped as a **single script/Make target** (never a two-line documented recipe — a forgotten step 2 is exactly the failure this project keeps hitting, and `run_ladder_perf_fpga.py:49` hardcodes an image name and never calls the freshness gate, so a half-run would silently measure the previous firmware).

Non-negotiables: keep `LD_LIBRARY_PATH=""` on the relink invocation (`Makefile:27,29,33,35` sets it on every buildroot call). **Do not** swap `Makefile:33` with `34-36` as a "permanent fix": `A=` is generic and `README.md` documents `A=linux-rebuild`, `A=modcapstone-rebuild`, `A=capstone-test-domains…` — packages that *feed* the rootfs and therefore *must* run before the cpio is rolled. opensbi is the sole exception because it *consumes* `images/`. A blanket swap relocates the wrong-binary bug onto `/test-domains/`.

Risk: none to freshness. It strictly strengthens the invariant — the firmware embeds the final Image by construction instead of by the accident that Image₁ == Image₂. `assert_firmware_embeds_current_initramfs` (`run_sqlite_baked_fpga.py:49-107`) still runs and still decompresses the embedded cpio; validate the reorder once against it, host-side, before any board time.

Bonus: the eliminated pass grows with the overlay (25.8 s → 36.5 s today, M), so R1 and R3 compound.

### R2 — Attach gdb to the wedge *inside* the run. **SAFE with three mandatory gates. Medium effort. ~200 s per wedge (I, bounded by M).**

`probe_sqlite_wedge.py:58-59` re-uploads and `cold_boot`s — 17.4 MB upload + 23 s power cycle + ~131 s JTAG + 81–142 s boot — purely to re-reach the state the preceding run was already in at `run_sqlite_baked_fpga.py:249`. The lock, console and boot are all still live there (there is no `except` today; `ActionTimeout` propagates to the outer `finally` at `:263-271`). Pairing is confirmed by content hashes: `board-O1.log` and its two follow-up probes ran the same `fw_<hash>.bin` and the same `initramfs 10490880 bytes` line; `upload_boot_image` (`fpga_console.py:527-530`) has no dedup, so a byte-identical 17.4 MB POST is repeated. Envelope M: board-linsafe 14:04 → probe-linsafe 14:10, minus the ~60–100 s gdb block that is retained ⇒ **~200 s**, not 250.

Precedent in-tree: `run_sqlite_stages_fpga.py:13-15` already argues exactly this, and `:87-92` already catches the timeout without re-booting.

Gates, all cheap, all required:

* **G-a — RE-RAISE.** `out` is already bound at `:218` to the `ls -l` text. If the new `except` swallows the timeout, control falls into `:250-262`, greps the `ls` output for markers, prints every marker MISSING plus a bogus `run took Ns`, and ends with `SQLITE ON SILICON: FAIL`. That relabels a wedge as a marker failure. Re-raise, or set an explicit WEDGE status.
* **G-b — `gdb_stop()` in a `finally`**, as `probe_sqlite_wedge.py:130-131` does. A leaked running session makes the *next* `cold_boot`'s `gdb_start` (`run_ladder_perf_fpga.py:155`) wait for a state event that never comes and burn a 60 s timeout. `release_board` does not stop gdb (`safe_cleanup.py:75-92`).
* **G-c — opt-in** (`SQLITE_PROBE_ON_WEDGE=1`), and wrap the probe so its own failure cannot mask the original exception. `run_sqlite_baked_fpga.py:129-138` documents a *false* wedge (a live run tripped at the 75 s idle limit mid-INSERT); default-on would tax those too.

Side benefit: the fold makes the probe inherit the on-board `sha256sum` STALE BOOT gate (`:201-215`) that `probe_sqlite_wedge.py` does not have. Freshness improves. Apply the same helper to `run_sqlite_stages_fpga.py:98-101`, which today prints "STOPPING: a wedged domain takes the core with it" and throws the live wedge away.

### R3 — Prune the overlay. **GATED. Medium. ~175 s/session (I from M fits) — the single largest lever, and the most dangerous.**

Measured cost of accumulation, from today's own logs: JTAG 128.9 KiB/s × (30.05 − 17.47 MB) = **96 s** (I), unpack 2.224 s/MB × (45.89 − 10.49 MB) = **79 s** (I). Nothing prunes: `benchmarks/sqlite/stage-sqlite-in-rootfs.sh:95-100` is a bare `cp -f` loop and the file contains no `rm`. The overlay now holds 28 files including `sqlite_lim1/lim512/nocapinit.dom` dated **Jul 30 21:39** — the very `INTERP_BUILD_LIMIT` diagnostics whose fake signature `stage-sqlite-in-rootfs.sh:74-83` records being "investigated as a hardware defect for hours".

**Why it is gated and not a recommendation.** Every variant `.dom` is *outside every freshness gate*:

* `run_sqlite_baked_fpga.py:94` iterates `for local in (LOCAL_HOST, LOCAL_DOM)`, fixed at `:112-113` to `sqlite_silicon.dom` and `sqlite_host.user`. No `sqlite_stageN.dom` is ever checked. The gate also `continue`s on a missing local file (`:95-96`) and only WARNs when no gzip member is found (`:83`).
* The on-board `sha256sum` gate is skipped for overrides (`:195-200`) and does not exist at all in `run_sqlite_stages_fpga.py` (no `hashlib`, no `sha256`).
* A pruned-away dom does not fail loudly. `{HOST} {dom}; echo DN_$?` gives `DN_127`, which **matches `r"DN_\d"`**, so `wedged=False`, `obs=None`, the first-bad test at `:120` never fires, and `:130` prints *"Every domain in this set returned rc=0 … The failure is outside what these stages cover"*. A confident false pass from a session that tested nothing — the exact class documented at `run_sqlite_baked_fpga.py:52-58`.

Today that hole is *structurally masked*: the overlay only grows, so a locally-present dom is necessarily in the firmware. Pruning removes the mask and leaves the hole open.

**G1 (blocking):** extend `assert_firmware_embeds_current_initramfs` to take the full list of binaries the runner will invoke (every entry of `DOMS` plus the effective `HOST`) and search each inside the cpio it already decompresses; raise instead of warn when no gzip member is found. Host-side, board-free, catches precisely the failure the prune introduces.
**G2 (blocking):** in `run_sqlite_stages_fpga.py`, treat `not found` / `DN_127` / `obs=None` as a HARD STOP with its own verdict, never as "no marker". Keep an explicit opt-out for the deliberate-nonexistent-dom probe at `run_sqlite_baked_fpga.py:115-118`.
**G3:** the wipe is its own explicit step, **never** inside `stage-sqlite-in-rootfs.sh` (that script is also invoked standalone after `run-sqlite-silicon.sh`, `:42-44`, and would silently delete a batch a previous step prepared).
**G4:** derive the keep-set mechanically from the runners' hardcoded defaults (`run_sqlite_stages_fpga.py:37-39`, `run_sqlite_baked_fpga.py:121`) plus the session's env overrides — **never** from what the logs show was executed. `sqlite_stage3.dom` appears in zero board logs and is a hardcoded default.
**G5:** keep the batch. `CLAUDE.md:119-122` mandates baking N variants into one initramfs; `board-refix.log` runs 17 doms in one boot. Prune *previous* batches and provably-dead files, not the current one, and keep deliberate control pairs (`stage14`/`stage14ctl`, `stage2/3/10` vs `s2fix/s3fix/s10fix`).

**Ungated interim that is safe today:** drop only the three Jul-30 `lim1`/`lim512`/`nocapinit` doms (4.87 MB raw) as a piggyback on a firmware rebuild already required for another reason. ~12 s of JTAG (I). Small, but it costs nothing and stops the worst stale-artifact bait.

Add the regrowth tripwire: fail loudly if `rootfs.cpio` exceeds ~12 MB.

### R4 — Move the HTTPS upload out of the lock. **GATED. Trivial. 20–40 s of board-locked time (I); 0 s wall-clock.**

`run_sqlite_baked_fpga.py:169-179` takes the lock, checks the bitstream, *then* uploads. The upload targets the console's HTTP image store, not the board (`config.py:152-157`: "No async state; the response is final"); the board-side copy is the separate JTAG `load_image`. Hoisting the POST above `console.lock()` removes it from the shared-resource window on every run, including changed-firmware runs.

**Gate:** only when `FPGA_FW_NAME` is unset **and** `IMG_NAME == _hash_name(IMG)` recomputed at call time. Under a fixed name (`run_ladder_perf_fpga.py:49` = `fw_payload_fpga_up_gpfree.bin`, `run-board-ladder.sh:70`) an out-of-lock write can overwrite an image a concurrent session is about to boot.
**Do this first, it is free:** wrap `:179` in a `time.time()` pair. The "30–90 s" figure is an unsourced docstring and the upload has never been measured on this link. Measure before optimising.

### R5 — `pr_warn_once` the SBI rfence spam. **GATED. Non-trivial. ~10 s/boot (M).**

620 identical `remote fence extension is not available in SBI v1.0` lines per live boot = **42,160 B of ~50,000 B** of live UART (M, `board-refix.log`), constant at 619/620 across a 4× initramfs range. Median inter-line delta **17.13 ms**, and the 587–591 deltas under 50 ms sum to **9.9 s** in both a fast and a slow boot (M) — effective console rate ~4 kB/s at 57600. `CONFIG_RISCV_SBI_V01` is **not set** (`build/build/linux-6.4.14/.config:290`), so `__sbi_rfence_v01` (`sbi.c:209-216`) is `pr_warn` + `return 0` with no ecall: the printk is the entire cost.

**Blockers, all real:**
* The only durable form is a **buildroot kernel patch**. `arch/riscv/kernel/sbi.c` lives under `sw/buildroot/build/build/linux-6.4.14/`, which `git check-ignore` resolves to `.gitignore:1:build/*`. An in-place edit is invisible to git, destroyed by `linux-dirclean`, and absent on the peer lane's clone. There is no `BR2_GLOBAL_PATCH_DIR` and no existing kernel patch — new plumbing, in a submodule we do not own (submodule protocol applies).
* **No gate covers the kernel.** `assert_firmware_embeds_current_initramfs` inspects only the decompressed initramfs; the STALE BOOT gate hashes only `/test-domains/` binaries. A firmware carrying an unpatched kernel passes both cleanly. Mitigating: a silently reverted patch makes the spam *return*, so the failure mode is a lost saving, not a wrong measurement.
* **Ring-buffer side effect.** Each boot currently burns ~52 KB of the 512 KB history ring (`socketio-api.md:74`), so it holds ~10 boots; at ~11 KB/boot it would hold ~48. `fpga_console.py:566` `reset(wait_prompt=True)` calls `wait_uart` with `search_from=0` against the prompt regex at `config.py:240` — it can already match a prompt replayed from a *prior* session. Latently wrong today, far more often wrong after this change. **Land the anchor fix first.**
* Do **not** pursue `loglevel=4` in `configs/fpgakernel.config:102`: `.config:302-305` has `CONFIG_CMDLINE_FALLBACK=y` and the live boot prints `Kernel command line: earlycon console=ttyS0,57600` — the DTS string from `configs/caplifive.dts:10`, not `CONFIG_CMDLINE`. It is a no-op that costs a rebuild.
* Diagnosability debit: those 620 lines are the only kernel output during ~27 s of userspace init and the densest liveness heartbeat in the window where a hang gets localised by hand. Pair `printk_once` with a single end-of-boot count.

### R6 — Doc hygiene. **SAFE. Trivial. 0 s directly; prevents wasted sessions.**

`HOW-TO-LAUNCH-ON-FPGA.md:71-80` still says same-VA batching is blocked. `ISSUES.md:499-511` marks R-3 `WORKED AROUND` since 2026-07-28 (address-keyed fault; distinct entry VAs; validated by a reversed-order control, spread 0.75%/0.03%, `instret` byte-identical) and `LADDER_DISTINCT_VA=1` / `LADDER_ONE_BOOT=1` shipped. Also stale: `:209` says the firmware is 17.4 MB (it is 30.0 MB). And note the counter-evidence that must stay attached: `board-stages789.log` shows `sqlite_stage10` wedging at position 4 of one boot, `board-stages.log` shows `stage2` wedging at position 3 while later stages 4/5/6 return cleanly elsewhere — **2 wedges in 14 same-VA runs (~14%)**. Same-VA reuse is intermittent, not safe; do not let the doc correction be read as clearance.

---

## 4. Rejected — do not re-propose

| Proposal | Why rejected |
|---|---|
| **JTAG-poke the `.dom` into a reserved DRAM window ("Route A")** instead of rebaking | Routes the artifact under test past *all three* gates (`assert_firmware_embeds_current_initramfs`, content-hashed image name, on-board `sha256sum`). Its replacement gate is a **D-side** digest of the window, while the known-open defect is **I-side**: `ISSUES.md:528-529`, "the monitor still lacks the icache invalidate on domain switch" — correct bytes read, stale instruction lines fetched, sha passes, previous binary runs. Also destroys the `exit 127` / `not found` fingerprint that today distinguishes a delivery failure from a domain bug. And the premise is wrong: batching is already worked around, so the baseline is ~85–115 s/dom, not 353 s. Needs a host-loader change, a DTS reserved-memory change in a foreign submodule, and a board-owner address. |
| **Swap `Makefile:33` with `34-36`** as the permanent ordering fix | `A=` is generic. `A=modcapstone-rebuild` installs `capstone.ko` into `TARGET_DIR` and `A=capstone-test-domains` installs `*.dom` into `TARGET_DIR/test-domains/` — both must precede cpio generation. The swap fixes opensbi and silently ships stale `.ko`/`.dom`, i.e. relocates the wrong-binary bug onto the exact path this driver runs. Fix opensbi-specifically (R1). |
| **Raise the OpenOCD/JTAG adapter clock** | `monitor verify_image` reads back over the same link, so at 2× the mandated verification cancels the entire saving (2×131/k; k=2 ⇒ 0 s). Without verification a flipped bit in OpenSBI/monitor text — the code under active development, and the ~9% of the image no gate hashes — presents as an intermittent M-mode wedge, indistinguishable from the bug class that already consumed six sessions. Bottleneck is unproven anyway: 128 KiB/s ≈ 30 µs/word fits USB/DMI round-trip latency as well as TCK. Asking the board owner what `adapter speed` is set to is harmless; acting on it is not. |
| **Raise the UART baud to 115200** | 25 MHz/16 = 1,562,500; /115200 = 13.56 — not divisor-exact, ±3–4% error, on a link that already needs sha-retry ladders and tolerant marker regexes (`config.py:242-243`). And the win is ~0.9 s once R5 removes the rfence bytes, not 5 s — the two proposals double-count the same bytes. Six in-repo baud sites, one of them (`sw/buildroot/caplifive.dts`) a stale duplicate. |
| **Skip the HTTPS upload when the stored image name already exists** | `IMG_NAME` is a content hash **only** in `run_sqlite_baked_fpga.py:46` and only when `FPGA_FW_NAME` is unset; `run_ladder_perf_fpga.py:49` and `run_sqlite_fpga.py:38` use a fixed name. A name-existence skip there permanently reuses a stale stored image — exactly the 2026-07-30 incident recorded at `run_ladder_perf_fpga.py:143-151` (booted a July-19 initramfs, `exit 127`, read as a domain failure, cost a session). Measured hit rate for the *safe* hash-name case is 2/12 today (~17%), all probe re-runs, never a rebuild iteration. Do R4 instead. |
| **Runtime-selected stage inside one fat image** (`selector` via shared region) | The selector must travel the domain-entry / shared-region channel, which has an **open integrity failure on this exact board** (`sqlite_capstone_domain.c:451-456`, and the live stage-14 finding that a string literal arrives with byte 0 intact and the rest zero-filled). A corrupted-but-in-range selector runs stage 3 while the log says stage 13. It also degrades the `0x5A6E_ssrr` marker from a compile-time constant (an independent witness of *which build ran*) to an echo of the host's own input. Real saving ~15 s: `-O0` means the N variants are byte-identical except one immediate, and they already compile in parallel. |
| **Batch 8–12 variants per boot** (beyond what the last result justifies) | 5 of 5 ladder rounds today authored *new* probe code in response to the previous board result (`sqlite_capstone_domain.c:251-258, 283-290, 309-318, 342-349`; commits 13d0b7410dfe → 9089e67673b1 → 06598df8c39d), so the wider brackets did not exist in advance. `run_sqlite_stages_fpga.py:101` breaks on first wedge, discarding everything past it. And every added variant is an unverified, same-sized (1538952 B ×6 today) binary in the image. Batching 3→6 is defensible; go wider only after G1/G2. |
| **`initcall_debug` board session** for the 14.7 s silent pre-8250 gap | Two independent false-negative paths. (a) The documented fast relink uses `FW_FDT_PATH=/tmp/capstone/caplifive_extracted.dtb` — a 3111 B blob *extracted from the known-good firmware*, not compiled from `configs/caplifive.dts` — so the bootargs edit never reaches the board and the board boots identically. (b) `init/main.c:1179,1189` print at `KERN_DEBUG` (7) while `CONFIG_CONSOLE_LOGLEVEL_DEFAULT=7` suppresses level ≥ console_loglevel, so nothing prints without `ignore_loglevel`. Either yields "initcall_debug named nothing unusual" measured on a stale artifact. The candidate set can be enumerated **offline for free**: 10 initcalls between `kyber_init` and `serial8250_init` in `.initcall6.init`. Do that first; the board session is the last resort, and the saving is 14.7 s ÷ N under mandated batching. |
| **Prune the QEMU-side overlay for speed** | QEMU boots are not on the board critical path. Score 0 s. Hygiene only. |
| **Skip the unconditional `sqlite_host.user` / domain rebuild in staging by carrying a hand-passed `EXPECT_DOM_SHA`** | The hash proves *artifact == what QEMU ran*, not *artifact == current sources*. A sha pasted from an earlier session still matches the stale artifact on disk. Acceptable **only** if computed and consumed inside one wrapper process, never from a human or a file, with both domain and host shas carried, and `SKIP_BUILD=0` remaining the standalone default. Saving ~42 s, and it is host-side prep *before* the lock (`HOW-TO-LAUNCH-ON-FPGA.md:13-14`), so it does not reduce board occupancy. |

---

## 5. What stays slow, and why

| Floor | s | M/I | Why it cannot be removed here |
|---|---:|:---:|---|
| JTAG `load_image` (17.5 MB pruned) | 131 | M | Flat 127.7–130.1 KiB/s across 13 loads and 3 image sizes. Fixed-resource, but *which* resource is UNRESOLVED (TCK vs USB/DMI round-trip). Only lever is a smaller image (R3), not a faster link (§4). |
| Power cycle (8 + 15) | 23 | M | `cold_boot` docstring: "a warm `monitor reset halt` does NOT work here — the fw_payload OpenSBI does not re-enter cleanly from a soft reset (its one-time hart/DDR init is not re-runnable)". |
| Kernel init to `clk: Disabling` | 22.3 | M | Invariant across 9 boots and a 4× initramfs range. Contains a **14.69 s ± 0.04** silent gap between `io scheduler kyber registered` and `Serial: 8250/16550` reproducible since 2026-07-21 and flat from 3879K to 15931K of init memory (so *not* async initramfs decompression). Candidates are enumerable offline; do not spend a board session (§4). |
| Userspace init → shell prompt | 36.6 | M | Constant to ±0.2 s across 8 of 9 boots. |
| initramfs unpack, at the pruned 10.5 MB | 22.5 | M | 2.224 s/MB is causal (single silent `populate_rootfs` gap; `Freeing initrd memory` scales 5920K→15928K). Below ~10 MB there is nothing left to cut. |
| Amalgamation compile | 42 | M | One 9.5 MB TU at `-O0`, single-threaded. N variants already run in parallel (91 ms apart), so the *marginal* variant is ~0 — this is a per-round floor, not a per-variant one. |
| `Image` regeneration on any `.dom` change | 30 | M | The initramfs is `CONFIG_INITRAMFS_SOURCE` built into the kernel. No domain-only shortcut exists that survives the freshness gates. |
| Board serialization | — | — | One physical board, one flock (`safe_cleanup.py:6-7`), secret token, human in the loop. Sessions cannot overlap; the only lever is making each one shorter. |

**Irreducible iteration ≈ 82 s host + ~330 s board ≈ 410 s (~7 min)**, of which ~250 s is JTAG + boot. Everything below that requires either a smaller firmware than the pruned baseline, a faster physical link, or a delivery path that bypasses the firmware — and the third is rejected on instruction-side staleness grounds (§4).

**Do first, in order:** R1 (safe, ~32 s, trivial) → R2 with G-a/G-b/G-c (safe, ~200 s per wedge) → R4's free measurement → then G1+G2, and only then R3, which is worth ~175 s/session and is the largest lever on the board critical path.