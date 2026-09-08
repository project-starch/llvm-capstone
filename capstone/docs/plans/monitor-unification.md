# Monitor-stack unification: one source per repo, target by build configuration

*Started 2026-09-07. Status lines below are updated as phases land. The full design and the
reasoning are in this file; the session-local plan it was written from is not a repo artifact.*

## Why

Three shared repos were checked out twice (the monitor three times) in this tree with different
content, and the two lines had not shared a commit since January 2025:

| repo | QEMU flavour (`capstone/caplifive-buildroot/…`) | board flavour (`capstone/caplifive-system/sw/buildroot/…`) | fork point |
|---|---|---|---|
| `caplifive-buildroot.git` | `capstone-bootstrap` 4d97ecf | `capstone-bootstrap-dts-65536` 5764378 | 2f56383 |
| OpenSBI wrapper (`capstone-opensbi.git` = `caplifive-opensbi`) | `capstone-bootstrap-qemu` 1048a61 | `capstone-bootstrap` 0ac0290 | 769939a |
| `capstone-sbi.git` (the monitor) | `capstone-bootstrap-qemu` e1ccb49 | `capstone-bootstrap-board` 56dfbe9 | 2f772bb |
| `caplifive-sbi.git` (the `sbi.dom` copy) | `capstone-bootstrap` 977af95 | (99aaffa) | 04ac643 |

**The split was ours, not the author's.** The buildroot author's own line, `fpga-qemu-integration`
(8c5518d), already builds both targets (`DEFCONFIG ?=`, "Fixed Makefile and kernel config for qemu
build"). We created `capstone-bootstrap` on 2026-05-07 from a January-2025 point behind even upstream
`master`, and `capstone-bootstrap-dts-65536` on 2026-08-05 directly on 8c5518d. Every fix since has
been ported by hand (Q-03 twice; Q-05 and the loader features never reached the board; the module's
fixes never reached QEMU). The 2025 bring-up hacks in the monitor (`rdtime` emulation, the
`split_out_cap` clone, `fence.i`) are the third contributor's; the author curated the line in 2026 and
re-added `fence.i` deliberately (99aaffa).

**Goal:** one branch in every repo (named `capstone-bootstrap-unified` during the merge, `capstone-bootstrap` since 2026-09-08 — see "Final scheme" at the end); `TARGET=fpga|qemu` selects
everything target-specific; both checkouts stay where they are and simply track the same branch.

## Safety record

- Tags by SHA on every pre-unify tip (`pre-unify/2026-09-08/<repo>-<flavour>`): buildroot-qemu
  4d97ecf, opensbi-qemu 1048a61, monitor-qemu e1ccb49, sbidom-qemu 977af95, sysdev 28f9436,
  buildroot-board 5764378, opensbi-board 0ac0290, monitor-board 56dfbe9, parent-dev f5b8acfb0902.
  Pushed 2026-09-08 (all but parent-dev and sysdev, whose commits are remote branch history). All-refs bundles of each nested repo under
  `~/capstone-artifacts/unify-backups/`.
- Work happens in a detached worktree (`~/capstone-artifacts/unify/wt-buildroot`, branch
  `capstone-bootstrap-unified`, from 5764378) with real nested checkouts; the live checkouts are
  untouched until the final switch.
- Reference artifacts taken before any edit, same compiler (`capstone/capstone-c` 8cda52c, which
  built every validated firmware on both sides): board `sbi_capstone_dom.c.S` dd549e86…,
  `capstone_int_handler.c.S` 55500a31…, `fw_jump.elf` f7c91efb…, `fw_payload.bin` 44c88d9e… (boot
  sw30); QEMU `sbi_capstone_dom.c.S` 64babc08…, `capstone_int_handler.c.S` 19fe1e3a…, `fw_jump.elf`
  d9b11509… (the Q-05 validation build).

## Phase A0 — buildroot: `TARGET`, one output directory per target, the loader merge

Commits c92bd54 (structure), 721cfbc (modcapstone merge) and 3361949 (the `sbi.dom` package copy follows the QEMU line) on `capstone-bootstrap-unified` (caplifive-buildroot.git, base 5764378). What changed and why is in their messages; the gates:

| gate | reading |
|---|---|
| FPGA `.c.S` pair regenerated from the unified tree with `-DCAPSTONE_TARGET_FPGA` vs the reference | **identical** (141030 B / 6618 B), 2026-09-07 18:38 and again after the full build |
| FPGA `fw_jump.elf` disassembly vs reference | **identical** (the only differing lines are objdump's file-path header) |
| FPGA `fw_payload.elf` `.text`/`.rodata`/`.data` vs the sw30 reference (44c88d9ebeb1) | **identical** (post-merge audit: also the `fw_jump` loadable image and both symbol tables byte-identical); the `.payload` section differs (17 % of its bytes — the embedded kernel+initramfs, whose contents changed: rebuilt kernel, merged module, the QEMU-line `sbi.dom`, and the board line's own extra test domains that a fresh build installs, +1.05 MB of initramfs, ~8 s per JTAG upload); `make_hole` present |
| fresh `build-fpga/` from nothing | first attempt failed: a payload firmware needs `images/Image` before OpenSBI links; fixed in c92bd54 (`build` builds the kernel first when `LINUX_PAYLOAD=1` and the image is absent); second attempt rc 0/0, zero make errors |
| QEMU target from `build-qemu` (fresh, rc 0, zero make errors): `.c.S` pair vs the QEMU reference | **identical** |
| QEMU `fw_jump.elf` disassembly vs reference (d9b11509…) | **identical** (path header only); new ELF 4d0abe120811 |
| C-11 control: fpga then qemu built in one tree | `fw_fdt_bin` symbols in the QEMU `fw_jump.elf` = 0; `.rodata` 0x50000 = reference; `build-fpga/` and `build-qemu/` coexist |
| QEMU chain from the unified images (`build-qemu`, lock held, own tmp root) | smoke rc 0; borrow-cost, revoke-cost, hier-revoke, revoke-on-free, intra-domain-mrev-revoke all green; the REV_TRANSFERRED hole fired 24 times, `0xdeadbeef` 0; SLT select1 1031 records identical; **child-share cascade probe hung at its round-2 read** on the unified rootfs and trapped on the live one — bisected (kernel, firmware and module swapped in turn, module byte-identical; page-allocator perturbation ruled out layout noise) to the board line's pin of the `sbi.dom` package copy (May-2026 monitor, no hole handling); fixed by following the QEMU line's pin 977af95 + wrapper (`sbi.dom` byte-identical to the live one), probe traps, smoke green |

Deliberately not touched in A0 (each is a Phase B item): the other packages' board-only edits
(`PERM_OUT→INOUT`, forced debug flag, `C_PRINT` guards, `__linear` annotations), the kernel choice
per target, the `sbi.dom` copy.

## Phase A1 — the monitor: one source, target `#ifdef`s (DONE in the worktree, 2026-09-07 19:41)

Commits (all gated by `gate-fpga-cs.sh` + `gate-fpga-fw.sh` = both FPGA `.c.S` files and the
fw_jump/fw_payload disassembly byte-identical to the pre-unify references, and `gate-qemu-cs.sh` =
the QEMU wrapper compiles over the unified monitor):

| repo | commit | step |
|---|---|---|
| capstone-sbi (monitor) | 835f15a | dead text deleted (the 325-line clone, duplicate define, stale comment, `.S` debug block) |
| capstone-sbi | 17d7b60 | `capstone_target.h`: per-target constants, macros only, `#error` without a target |
| capstone-sbi | d3adf3f | reporting layer per target: UART block FPGA-only; QEMU reports = trace-print pairs |
| capstone-sbi | 1b87df8 (was 9b5fc28 before the post-merge audit's message amend) | the E/C hunks per target verbatim (whole functions: `print_regions`, `print_cpmps`, `make_hole`, `create_domain`, `handle_exception`; inline: SPLA/SPLB reports, `REV_TRANSFERRED`, `swap_cpmp` prints, IRQX; QEMU-only pre-checks; `MAKE_HOLE` call-site macro; the `.S` fence.i/reentry/rdtime FPGA-only) |
| opensbi wrapper | 86b5f92, ea34f91 (175206f before the audit amend: the spurious QEMU `csrw mscratch` removed) | `platform/fpga/ariane` defines the FPGA target; the wrapper per target (UART mint, fence.i, MIE, stack size, int-handler `capstone_error`, `fw_base.ldS` ALIGN); `platform/generic` defines the QEMU target + `CAPSTONE_DEBUG_ENABLE` |
| caplifive-buildroot | 6cced67, then 21b09c2 (audit follow-ups) and 41a2a1a (ignore fix) | one `components/opensbi` for both targets; the `.c.S` regenerates on a stamp of the defines when TARGET changes (positive control: after a QEMU build, `make -n TARGET=fpga` regenerates) |

QEMU side after the collapse: the `.c.S` regenerated through the real Makefile equals the
compile-check output exactly; it differs from the QEMU reference in ten functions, all enumerated in
9b5fc28's message (tagged reports where the QEMU line printed legacy codes, `capstone_error`
printing the CERR tag after `0xdeadbeef`, the spec-correct `swap_cpmp` read-back). Nothing else moved.

**QEMU chain on the unified firmware (fw_jump bbe6d434b12c, rootfs 96ada54aaf29, 2026-09-07 19:41–19:46):**
smoke rc 0; borrow-cost, revoke-cost, hier-revoke, revoke-on-free, intra-domain-mrev-revoke green
(the REV_TRANSFERRED hole fired 24 times, `0xdeadbeef` 0); the child-share cascade probe TRAPPED;
SLT `select1` 1031 records, 1000 queries, `completed=1`. Module-consistency check
(`tests/runtime-qemu/run-q03-region-hole-check.sh`, new: 8 domains through the check loader, the last
one `chk*`): 8/8 returned, `count=9`, fresh region `qlen=4096 mmap=ok write=1`, `oob_share
retval=4294967295`, `reuse=0 fetchfail=0`, `__CAPSTONE_Q03_REGION_HOLE_CHECK_PASSED__` (20:08). Corpus tier
(corpus-runner, lock held 19:46–20:07, every QEMU invocation verified to use the unified images):
authority 32/32, RV8 7/7 (sha512 passed on a boot-flake retry), BEEBS 80/81 — `nettle-sha256` hung
once with no serial output after its command (N=1); redrawn 20:10: unified 2/2 PASS, live 1/1 PASS —
a run flake, not the images; BEEBS therefore 81/81 counting the redraws.

**Post-merge audit (claim-auditor, 2026-09-07 20:11):** FPGA side CONFIRMED beyond the gates (`cpp
-DCAPSTONE_TARGET_FPGA` of the unified monitor vs 56dfbe9 differs in ONE line, `make_hole((i))`; the
`.S` only in the deleted comment block; fw_jump loadable image and both symbol tables byte-identical;
the FPGA define reaches the assembler and the linker script — three `ALIGN(0x1000)`, `dom_stack`
0x2000). QEMU side: no FPGA-only code leaks (0 fence.i, 0 UART, 0 rdtime in the QEMU preprocessed
text; the QEMU `.S` byte-identical to e1ccb49), but one spurious instruction was found — a
`csrw mscratch, s2` I had added to the QEMU arm of `sbi_capstone_init.S` that neither line ever had
active — **removed**, wrapper commit amended (ea34f91); three inert differences added to the
enumeration (1b87df8). Also found: the worktree's `build-fpga` still held the pre-fix `sbi.dom`
package copy (buildroot does not re-sync a local package on a bare pass) — rebuilt, and the board
bake script now rebuilds the two moved packages explicitly; the regeneration stamp was rewritten by
`make -n` — guarded (21b09c2); the `.c.S` is now written atomically; the board's `memset` of the
share-ioctl argument struct is kept (21b09c2).

**Re-validation after the audit corrections (fw_jump ec508f14bbb6, the QEMU arm of the wrapper now
byte-for-byte the QEMU line's, module with the memset):** the first chain run hit three infrastructure-
shaped failures (QEMU exiting as the smoke command was typed; two kernel boot stalls before login — a
shape present in live-tree corpus logs too — and a timeout) while the five probes passed; redrawn on
the quiet host: smoke 2/2, SLT `select1` 1031 records `completed=1`, child-share 3/3 TRAPPED (its
two stalls were both before login, i.e. before the probe ran). **Which gate ran on which firmware:**
the corpus tier (authority, RV8, BEEBS) and the module-consistency check ran on bbe6d434b12c, before
the audit corrections; only smoke, the five probes, child-share and SLT `select1` re-ran on
ec508f14bbb6. The corrections moved the tree toward the originals, so the risk was low, but the
statement "every gate green on the firmware the tree builds now" held only after the live-checkout
re-run below.

### Design as executed

Base: 56dfbe9 (board). Principle: the FPGA target's two generated `.c.S` files stay byte-identical
through every commit; the QEMU target's differences are enumerated per commit. Conflicting hunks are
carried verbatim under `#ifdef CAPSTONE_TARGET_FPGA` / `CAPSTONE_TARGET_QEMU`; that is scaffolding,
shrunk in Phase B.

**Rules for the FPGA branch, each measured by the pre-A1 audit against the reference hash
(`capstone-c` 8cda52c, `lang.rs:346-368`, `codegen.rs:1784-1811`):**
- Comments, blank lines, `#define`s, `#ifdef` blocks at file scope or inside a function, and whole
  functions excluded by `#ifdef`: **no change** to the output.
- **Any new declaration changes the output**: a new global (a `.gct` slot; inserted early it shifts
  every `gp` offset — 450 lines), a bare prototype, an `extern` (both are treated as global
  definitions: same hash as the unused-global control), a `static inline` helper (a body is emitted).
  So: no new global, no prototype, no `extern`, no `static inline`; FPGA-side helpers are macros.
- **A changed function signature changes the output**: `make_hole(unsigned i, unsigned tag)` with
  `tag` unused grows the frame 144→160 and shifts every spill offset, and the caller changes. The
  FPGA signature stays `make_hole(i)`; the arity lives in a call-site macro
  `MAKE_HOLE(i, tag)` → `make_hole((i))` on FPGA, `make_hole((i),(tag))` on QEMU (verified: FPGA
  hash unchanged).
- The gate is `~/capstone-artifacts/unify/gate-fpga-cs.sh`: deletes both `.c.S`, regenerates them
  through the Makefile's **absolute** targets with the FPGA defines, checks the exit status, then
  `cmp`s against the references. Negative-tested 2026-09-07 19:12: a flipped token → DIFFERS (58
  lines); an unreferenced global → DIFFERS (8 lines); restored source → IDENTICAL. (The first gate
  script used relative target names against absolute rules and never fired — its 18:38 "identical"
  reading was true only because `make build` had regenerated the files as a side effect.)

### Hunk classification (board 56dfbe9 vs QEMU e1ccb49; line numbers board/QEMU)

Categories: **A** target configuration · **B** pure addition, correct on both · **C** hardware-behaviour
hack needing a switch · **D** dead/superseded · **E** genuine conflict.

`sbi_capstone.c`

| # | board | QEMU | what | cat | A1 disposition |
|---|---|---|---|---|---|
| C1 | 183 | 19 | `C_PRINT` = `csrw 0x800` vs `.insn r 0x5b,0x1,0x43` | A | target header |
| C2 | 20–166 | – | tag/error-code block, `capstone_trace`, `capstone_error_tag`, `CAPSTONE_SHARE_TRACE_ENABLE` | B | shared (defines emit nothing) |
| C3 | 166 | 28 | `capstone_error` → tagged report vs bare `C_PRINT` pair | B | reporting macro |
| C4 | 177–183 | 30 | `#ifdef CAPSTONE_DEBUG_ENABLE` guard on the debug counters (board) vs unconditional (QEMU) | C | guard everywhere; define only for QEMU |
| C5 | 190 | 36 | `CSR_TIME` define (rdtime unit) | C | FPGA-only |
| C6 | 222–341 | – | UART reporting block (`capstone_uart_ready`, `capstone_report`, …) | **C** (audit) | FPGA-only. On QEMU virt the UART at 0x10000000 has 1-byte register spacing and 8 mapped bytes; the monitor reads LSR at +20 → unassigned → load access fault, and `capstone_trace` is unconditional (`CAPSTONE_SHARE_TRACE_ENABLE`, `:112`), so C6 **with** the `cap_env_init` mint faults at the first `create_domain`; the mint also appends a region and shifts every QEMU region id. On QEMU the reporting macro maps to `C_PRINT` and the UART code is excluded. |
| C7 | 348–462 | 74–180 | `read/write_cpmp` whitespace cleanup | B | board text |
| C8 | 394–462 | 120–180 | cpmp default: `RCPX`/`WCPX` reports vs `C_PRINT(0x8888888/0x9999999)` | B | reporting macro |
| C9 | 469–471 | 187–196 | `print_regions`: board skips holes and has NO prints; QEMU prints, no hole skip | **D** (audit: 1b81b28 stripped the prints in the same hack commit that changed the print encoding — not a decision) | A1: FPGA body verbatim (byte identity); Phase B restores the prints as reports on both |
| C10 | 482–492 | 204–212 | `print_cpmps` prints stripped on the board | D | same as C9 |
| C11 | 498–509 | 222–234 | `make_hole(i)` + tagged reports vs `make_hole(i, tag)` + `C_PRINT` | E | per-target signature under `#ifdef`; `MAKE_HOLE(i, tag)` call-site macro (a shared two-arg signature is NOT free: audit E2a) |
| C12 | 538–547 | 263 | `split_out_cap` no-region report | B | reporting macro |
| C13 | 556–570 | 273–284 | non-linear exact-fit guard report; `make_hole` arity | B/E | as C11 |
| C14 | 587–623 | 306–330 | table-full reports in the two appends | B | reporting macro |
| C15 | 607–610 | 319–322 | `cap_type()`/`CAP_TYPE_LINEAR` vs `__capfield(…,1)`/`0` — numerically identical | B | board text |
| C16 | 634 | 340–353 | QEMU doc comment on `globals_off` | B | keep |
| C17 | 652 | 371–387 | `gpoff = GPFREE_GLOBALS_OFFSET` vs `gpoff = 0` | **E** | per-target block |
| C18 | 661–736 | 396 | C-13 representability rounding (board only; RTL `compress_bounds` granule) | C | FPGA-only |
| C19 | 726–735 | 396 | one declarator per line (board; capstone-c accumulates `*` across declarators) vs one line | E | per-target block (coupled to C24) |
| C20 | – | 396–400 | QEMU pre-carve `region_n + 1 > MAX` check in `create_domain` | B | `#ifdef` QEMU-only in A1 (B-item 1) |
| C21 | 736–737 | 398 | `DBAS`/`DENT` traces | B | shared, reporting macro |
| C22 | 741–742 | 400–401 | splits at `split_size`/`data_off` vs `code_size`/`DOMAIN_DATA_SIZE` | C | with C18 |
| C23 | 807–821 | 455–473 | blob-copy guards | E | per-target block |
| C24 | 855–856 | 490–491 | copy-loop unit `>>3` (words) vs `>>4` (caps) | **E** | per-target block |
| C25/C28 | 863–927 | – | `CAPSTONE_DOMAIN_TRAP_VECTOR` acceptance test, guarded OFF | B | keep (compiles to nothing) |
| C26/C27 | 879–889 | 510–527 | unconditional gp carve + cscratch slot vs conditional | **E** | per-target block |
| C29 | 943–956 | 545–548 | `ENT0/ENTB/ENT1/ENT2` markers | B | shared, reporting macro |
| C30 | 984–1000 | 570–576 | `create_region`: post-carve `RGNO` spin vs pre-check `return -1` | E | FPGA verbatim; QEMU pre-check `#ifdef` QEMU (B-item 1) |
| C31/C34/C35/C38/C39/C46 | various | various | `SHA*`, `DRET`, `ECSA…`, `DPI*` trace sites | B | shared, reporting macro |
| C32 | 1098 | 639 | `CAP_TYPE_LINEAR` vs `0` | B | board text |
| C33 | 1111–1141 | 652–666 | `REV_TRANSFERRED`: `SHAX` report + CPMP clear vs `make_hole(region_id, 0x1239)` | **E** | FPGA verbatim; QEMU hole `#ifdef` QEMU (B-item 2) |
| C36 | – | 711–716 | QEMU pre-check in `share_child_region` | B | `#ifdef` QEMU-only (B-item 1) |
| C37 | guards | guards | `region_live[id]==0` guards identical; comments differ | B | board text |
| C40 | 1581–1587 | 1077 | `IRQX`+`MCAU` on unhandled interrupt | B | shared |
| C41 | 1600–1603 | 1090–1091 | `swap_cpmp` read-back-write (c777322 "linearity issue") | **B** (audit: the QEMU direct form is a SPEC VIOLATION — `mem-access-insn.adoc:53-55`, `ldc` of a linear cap nulls the slot; QEMU masks it) | board text on both, no `#ifdef` |
| C42 | 1608–1622 | 1096–1101 | `swap_cpmp` no-region reporting | B | reporting macro |
| C43/C44 | 1650–1730 | 1129–1146 | `handle_exception` returns `time_val`; `rdtime`/mtime emulation; `ILLX` report | C (ILLX part B) | FPGA-only unit with S3 |
| C45 | 1692–1726 | 1140–1145 | `EXCX/MCAU/MEPC/MTVL/MSTA` on other exceptions | B | shared, reporting macro |
| C47 | 1796–2120 | – | 325 lines of commented-out `split_out_cap_a` | D | delete |

`sbi_capstone.h`: `CCSR_CATP/CDC` (B), `CAP_TYPE_*` named constants (B; the "different from the
spec" comment is stale — values equal QEMU's literals; deleted in A1 step 1), `PRINT` encoding (A), `SHRINK` (B),
`CSR_CIS` 0x804/0x800 (A), `CSR_OFFSETMMU`/`CSR_CDCB` (A), `CAPSTONE_MAX_DOM_N` 32/64 (E → per
target in A1, B-item 3), `CAPSTONE_MAX_REGION_N` 96/64 (E → per target in A1, B-item 3).

`sbi_capstone.S`: `_dom_reentry` reads `mepc`/`mcause` (B); seven `fence.i` (C, FPGA-only);
`_handle_non_ecall` rd write-back for the rdtime unit (C, FPGA-only, with C43/C44); the commented
`_get_smode_context` block (D, delete).

OpenSBI wrapper (both platforms build `platform/generic` identically; every `platform/fpga/ariane`
hunk is inert on QEMU): `sbi_capstone_dom.c` UART mint (B, paired with C6) and six `fence.i` (C);
`mstatus.MIE` left set (C, `sbi_capstone_dom.c` + `sbi_capstone_init.S`); `dom_stack` 8 KB vs
64 KB (E → per target); `capstone_int_handler.c` `capstone_error` neutered + counter guard (C);
`fw_base.ldS` `ALIGN(0x1000)` on `.cap_text` (C, reaches the QEMU firmware; reason unrecorded);
`fw_payload.S` `.fdt` section (A, `FW_PAYLOAD_FDT_PATH`); the generated `.c.S` (D, regenerate).

Not divergences (the record misstated them): cap-type numbering (reverted to 0/1, equal to QEMU's
literals); the kernel module's `MAX_REGION_N` (64 on both); the template-copy commit 04ac643
(both lines carry a superseding descendant).

### Formerly unresolved — settled by the pre-A1 audit (2026-09-07)
1. **Settled.** The declarator accumulation is compiler-wide (`lang.rs:346-368` mutates one
   `last_type` across declarators), so the split-declarator discipline and `>>3` are required under
   the QEMU build too — Phase B item 5 converges on the board form.
2. **Settled.** 1b81b28 stripped the prints in the same hack commit that changed the print encoding;
   not a decision. C9/C10 → D.
3. **Settled.** Four encodings exist, one live: `csrw 0x800` (`sbi_capstone.c:183`, six call sites);
   `.h:31 PRINT` unused; `capstone_int_handler.c:13` defined but never invoked; `.c:182` commented.
4. Partial: 1d1ffa8 lowered both table sizes 64→32; `MAX_REGION_N` was later raised to 96 without
   trouble; no evidence 32 is required for `MAX_DOM_N`.
5. **Unresolved.** `ALIGN(0x1000)` (opensbi c72c132 "correcting alignment", empty body). Cheapest
   settling step: run the committed `compress_bounds` model over `_cap_int_handler_text_start/end`
   at `ALIGN(8)` vs `ALIGN(0x1000)`. Kept for both targets in A1.
Also from the audit: C18's RTL citation is `ariane_pkg.sv:786-840`; the `.c.S` rule depends on the
sources but not on `TARGET` (safe only while the two targets use different wrapper directories — the
A1 collapse commit must add the defines as a prerequisite stamp); `make setup` on an initialised
tree would reset nested checkouts (guarded in c92bd54: init only when absent).

## Phase A2 — validate, audit, switch (DONE 2026-09-07 22:19)

**Board (boot sw31, `tests/board-results/2026-09-05.tsv`):** the live `caplifive-system/sw/buildroot`
checkout moved to the committed unified tree (21b09c2 / opensbi ea34f91 / monitor 1b87df8 / sbi.dom
package 977af95), `.config` regenerated from the new defconfig (override file `local-fpga.mk`
confirmed in it), the two moved packages rebuilt, the initramfs re-packed, the firmware relinked:
`.c.S` pair identical, `fw_jump` disassembly identical, `fw_payload` `.text` identical to sw30's;
`fw_payload b05bf97e8857` (17.5 MB with the SLT trio staged). ONE boot, eight domains: k800 = 4
first, six BEEBS rungs at their native oracles, SLT `select1` 1031 records / 1000 queries identical to
native, zero `SPLA/SPLB/RGNO/EXCX/CERR/ILLX/MCAU`, `HOLE` 0 (no exact fit occurred; the hole path
remains unexercised on silicon and self-reporting). The firmware is listed in
`tests/firmware-with-q03-hole.txt`; preflight C7 then passes 8 domains by content with no override.

**QEMU:** the worktree chain (A1's re-validation paragraph), then the **live-checkout re-run
(2026-09-07 22:31, `fw_jump 557b444b6bf3`, the firmware the live tree builds)**. First finding: after
the switch, the live QEMU checkout's `test-domains` and `nested-enclave` binaries were still the QEMU
line's while their sources were now the board line's — buildroot does not re-sync a local package on
a bare pass, the same trap the audit caught in `build-fpga`. Both packages rebuilt (`A=<pkg>-rebuild`);
all 23 files under `test-domains/`, `nested/`, `null_blk/` then byte-equal to the validated worktree
`build-qemu`; rootfs 7302d4e3c1f4. On that rootfs: smoke `retval = 42`; module-consistency check
8/8 returned, `count=9`, `oob_share retval=4294967295`, `reuse=0 fetchfail=0`, PASSED;
authority 32/32; RV8 7/7; BEEBS 81/81 with zero infrastructure retries (lock held 22:31–22:47,
per-run root `/tmp/capstone-live-reval`). Every QEMU gate is now green on the firmware the live tree
builds, from its own images.

**Not yet exercised on silicon: the merged loader library.** The A2 boot ran pre-built programs:
`lpc` (private structs, no libcapstone) and `sqlite_host_1m.user`, linked before the merge against
the QEMU line's library. The first rebuild of a board host program links the merged
`libcapstone.c`, which packs the `.capstone_gp_initdesc` offset into `entry_offset` and synthesizes
the image across all `PT_LOAD`s (the board line's loader spanned from the first executable `PT_LOAD`
to the last one). QEMU has validated that shape; the board has not — treat that rebuild as a
validation boot (control first), not routine. The board rootfs's nested-enclave images are
`miniweb_backend.smode` ET_EXEC with two `PT_LOAD`s and ET_REL `.ko` modules (`readelf -lW`,
2026-09-07); nothing in the A2 boot loaded them.

**Audits:** pre-A1 and post-merge, both folded in (see above).

**Switch:** the branch `capstone-bootstrap-unified` is checked out in both live checkouts (the worktree
`~/capstone-artifacts/unify/wt-buildroot` stays as a detached reference); the live QEMU checkout was
rebuilt on it and smoked. Pushed 2026-09-08 12:47 (every nested push a branch creation; the lead's terminal for the three repositories the agent credential cannot write); every gitlink in `dev` resolves on GitHub, verified afterwards.

### Plan as written

QEMU from `build-qemu` under the lock: smoke; the five probes; hole non-vacuity; SLT `select1`;
the QEMU core tier; the child-share probe; the module-consistency check; manifest B or the twin
suite. Board: the live `sw/buildroot` checkout switched to the committed unified state, one boot,
k800 first, six BEEBS rungs, SLT `select1` last. Two audits (before A1: this table and the
byte-identity claim; after A2: preprocessed-source equivalence and the readings). Then docs, the
switch of the live checkouts, and the pushes (done 2026-09-08; every push was a branch creation).

## Phase B — convergence backlog (each its own commit and gate)
1. QEMU's three pre-carve bounds checks on FPGA, replacing the post-carve spin — board boot.
2. ~~Q-05 `make_hole` at `REV_TRANSFERRED` on FPGA — board boot with a transfer-annotated share.~~ **DONE 2026-09-08** (boots sw36/sw37).
3. Geometry (`MAX_REGION_N` 96 / `MAX_DOM_N`) on QEMU — makes M-2 reachable under QEMU.
4. The other packages both ways (the QEMU line's `__linear` annotations and `C_PRINT` guards into
   test-domains / null-blk / nested-enclave; the unified tree carries the board line's versions and
   both live rootfs images are built from them since 22:31).
5. The gp cluster as one logic (carve only when the image declares globals; split declarators +
   `>>3`) — QEMU tier + full ladder board boot. Highest risk; last.
6. `swap_cpmp` read-back and the C-13 rounding on QEMU.
7. ~~The `sbi.dom` copy onto the unified monitor source.~~ **DONE 2026-09-08** (see "Final scheme").
8. `fence.i` / `rdtime` / `mstatus.MIE`: rtl-oracle on the current RTL, then one firmware-only board
   boot without `fence.i`. The `#ifdef` stays until that boot passes.
9. M-2: bound the module's region copy before any >8-domain SQLite boot.
10. Kernel unification (the lead's later decision).

## Final scheme (2026-09-08): one branch name, the sbi.dom copy retired

**Why a second step.** After A2 every repository had a new name, `capstone-bootstrap-unified`, next
to its old ones; the lead asked for a permanent scheme. The old `capstone-bootstrap` tips were
siblings of the unified line (the QEMU buildroot line, the QEMU wrapper line, the sbi.dom monitor
line), so the name could not move onto it without a force-push. Instead each repository got a
**lineage merge**: `git merge -s ours` of the old line(s) into the unified tip — the tree stays the
unified one, the old tips become ancestors, and `capstone-bootstrap` fast-forwards onto it everywhere
(hook-compatible, nothing rewritten). Monitor 3da7ebe (parents 1b87df8 ← e1ccb49, then ← 977af95),
wrapper 3de3342 (ea34f91 ← 1048a61, monitor pin 3da7ebe), buildroot b7fc740 (41a2a1a ← 4d97ecf,
pins 3de3342 / 3da7ebe). The QEMU monitor and wrapper lines are carried in content (A1's gates); the
QEMU buildroot line's package edits are NOT (item 4) and b7fc740's message says which — lineage is
not presence there.

**Item 7 folded in, because it is what makes the monitor repository one-branch.** The sbi.dom
package (`package/capstone-sbi-domain`) now `#define CAPSTONE_TARGET_QEMU` before including the
monitor source and pins the same commit as `components/opensbi/lib/sbi/capstone-sbi`; the separate
977af95 copy (04ac643's template copy + Q-03/Q-05 cherry-picks) is superseded. Generated sbi.dom
from the unified source: 6259-line `.c.S`, no FPGA-only text (0 `fence.i`, 0 UART CSR writes), QEMU
print encoding, `make_hole` present. QEMU firmware unchanged (fw_jump 557b444b6bf3 live /
ec508f14bbb6 worktree).

**Validation on the final tips (2026-09-08 13:23–13:40).** Worktree images (`/tmp/capstone-unify`):
smoke 42; borrow-cost, revoke-cost, hier-revoke, revoke-on-free, intra-domain-mrev green; child-share
cascade TRAPPED (sbi.dom's only consumer); SLT select1 1031 records / 1000 queries `completed=1`;
module-consistency check 8/8, `count=9`, `oob_share retval=4294967295`, `reuse=0 fetchfail=0`. FPGA:
`.c.S` pair IDENTICAL, `fw_jump` IDENTICAL, `fw_payload .text` IDENTICAL to boot sw31's references.
Live QEMU images rebuilt: fw_jump unchanged, sbi.dom byte-equal to the worktree's, smoke 42.
Board boot sw32 (2026-09-08 13:43–14:00, `fw_payload be14f318a56f`, 17.5 MB, listed in
`tests/firmware-with-q03-hole.txt`): k800 = 4 first, six BEEBS rungs at their native oracles, SLT
select1 1031 records / 1000 queries identical to native, zero `SPLA/SPLB/RGNO/EXCX/CERR/ILLX/MCAU`,
`HOLE` 0. The SQLite host program of that boot was rebuilt against the MERGED loader library
(8e52da5a1793, marker strings present) and validated under QEMU on select2 first — the first run of
the merged library on silicon, closing the caveat recorded above. QEMU nightly tier on the live
images (2026-09-08, `--skip-build`, `parent c20716d54f2f`): 17 of 18 suites PASS on the first run;
the 18th, `linear-uninit-corpus`, failed on `linear_drop_sibling_ok`. Run down and fixed the same
day: bisected to the Q-05 monitor commit (a pre-existing stale host-observer read the unification's
nightly surfaced, not a merge defect; ISSUES.md Q-05, 2026-09-08 note), fixed in the corpus
controller to read back through the domain's alias, suite then green 3/3 at O0/O1/O2. Tier **18/18**.

**Local checkouts.** Both trees on local branch `capstone-bootstrap` at these tips (all levels, plus
the sbi.dom package checkout); the local `-unified` names deleted; the board tree keeps its old
`capstone-bootstrap-dts-*` local names (frozen, ancestors). The worktree
`~/capstone-artifacts/unify/wt-buildroot` holds the same tips on scratch branches `unify-final`.

## Phase B — decision memo (2026-09-08, investigated; the lead's calls marked ◆)

Facts gathered from the unified tree (`sbi_capstone.c` line numbers as of monitor 3da7ebe): the
monitor still has 23 per-target blocks — 3 whole functions (`make_hole`, `create_domain`,
`handle_exception`), 2 print bodies, 11 `fence.i` sites in `.S`/`_dom.c`, the pre-carve checks, the
REV_TRANSFERRED site, the reporting layer, and 3 link/stack/MIE details. Each item below is one
commit with its own gate; "boot" means a board boot with control first.

| # | item | what it changes | pro | con / risk | cost | recommendation |
|---|---|---|---|---|---|---|
| 1 | QEMU pre-carve slot checks on FPGA (`:1269`, `:1505` vs the board's post-carve `SPLA`/`RGNO` spin `:610`) | a full table refuses the share/carve with `-1` BEFORE splitting, instead of splitting first and spinning with a report | closes a real board wedge shape (a carve that half-completes at a full table); already proven on QEMU | changes error behaviour the drivers classify (`SPLA`/`RGNO` lines stop appearing; a refused share returns -1 to the host) | 1 boot: a load that fills the table (M-2 first, see 9) | **do, after 9**; low risk, one gate |
| 2 | Q-05 hole at REV_TRANSFERRED on FPGA (`:1402`) | the transferred slot becomes a hole; the board today clears only the CPMP mapping and keeps the slot | removes the documented silent-wedge shape (host access after a transfer reaches `cap_base` on a nulled slot) | no board probe exercises a transfer-annotated share yet — the boot needs a new domain (port `held_no_revoke_ok` + the read-back-through-domain observer) | 1 boot, new probe | **do**; the observer design is done (Q-05 close-out) |
| 3 | geometry on QEMU: `MAX_REGION_N` 64→96, `MAX_DOM_N` 64→32 (`capstone_target.h:27-34`) | one table size everywhere | makes M-2 reproducible OFF the board (96 monitor slots > the module's 64) — that is how item 9 gets a QEMU positive control | `MAX_DOM_N` 32 could bite a QEMU suite that creates >32 domains in one boot (BEEBS boots 82 domains serially — one at a time, fine; manifest B 24) | QEMU tier only | **do first**; it is the prerequisite for testing 9 |
| 4 | the QEMU line's package edits (20 files: nested-enclave, null-blk, test-domains, dpdk lib, miniweb) | carry `__linear` annotations, `PERM_OUT→INOUT`, `C_PRINT` guards both ways | one source of truth for the packages | only null-blk has a gate (`nullblk-all` in the nightly); nested-enclave and dpdk have none — carrying unverified hunks is the "no gate" trap | QEMU tier (null-blk); the rest is code review only | **defer** except null-blk; do nested/dpdk only when a suite exists |
| 5 | the gp cluster as ONE `create_domain` (306 FPGA vs 247 QEMU lines: `gpoff` fallback, `>>3`/`>>4` declarator workaround, blob guards, cscratch slot) | one domain-creation logic | the biggest remaining duplicate; every future loader change is done twice until then | highest risk in the list: it is the code that wedged silicon for weeks (gp delivery); the `>>3`/`>>4` split is a documented capstone-c codegen quirk on the board build | full ladder boot + QEMU tier; byte-identity gate cannot help (behaviour changes by design) | **last**, and only with the pre-registered readings written down first |
| 6 | C-13 rounding and the `swap_cpmp` read-back on QEMU (`:741`, `:898` DIAGNOSTIC MAGIC) | apply the board's representability rounding on QEMU too; keep or drop the C-13 diagnostic word | rounding is spec-correct on both; QEMU would then reject the same images the board rejects | the C-13 magic is marked "REMOVE once C-13 is closed" — C-13 IS closed (R-10) → it is dead weight on the board | QEMU tier; removing the magic needs one boot (it is written into every image) | **do**: rounding → shared; magic → remove with the next board boot |
| 7 | sbi.dom from the one source | — | — | — | — | **done 2026-09-08** |
| 8 | `fence.i` ×11 (FPGA-only), `rdtime` emulation (FPGA-only), `mstatus.MIE` clear/re-enable (QEMU-only — the earlier note had this inverted) | drop what silicon no longer needs | fewer per-target blocks; `fence.i` on every split is a measurable cost | wrong removal = a silent icache-stale wedge with no print | rtl-oracle read 2026-09-08 (below), then one firmware-only boot | **rdtime: KEEP, permanent** — the RTL has no `time`/`timeh` CSR (`csr_regfile.sv:694-709` has no case; `default: read_access_exception` → `ILLEGAL_INSTR`, any privilege), QEMU implements it (`csr.c:4144`); this is a silicon fact, not a hack. **fence.i: UNRESOLVED** — neither a fetch-side capability cache (CPMP feeds only `ex_stage`/`load_store_unit`, never the frontend) nor ordinary icache non-coherence (`split_out_cap` writes no code bytes) explains it; the surviving candidate is that Capstone `CCSRRW` writes never assert `flush_o` while ordinary side-effecting CSR writes do (`csr_regfile.sv:2374-2550` vs `:1160-1535`, `controller.sv:209`), and only 3 of the 9 `.S` sites sit after a `CCSRRW`. Second read (ISSUES.md **R-26**): the three `.S` sites after a `CCSRRW` to `cscratch`/`cpmp` ARE load-bearing — those writes land at commit with no `flush_o`, and LDC/STC/CALL/RETURN issue with no interlock and read `cpmp_q`/`cscratch_q` combinationally; keep them. The other eight (`cap_env_init`'s five after split/store, three more in `.S`) remain unexplained → ONE firmware-only boot with two variants staged (keep only the 3 CCSRRW-adjacent / keep only the `cap_env_init` ones), control first. Until then keep all 11. **MIE: leave** — QEMU clears it at capmode entry and re-enables it after `cap_env_init`; FPGA inherits OpenSBI's value; no RTL reads `mstatus`; no silicon claim either way. |
| 9 ◆ | M-2: the module copies the monitor's region table into `regions[64]` with no bound (`capstone.c:30,:195-`) | either bound the copy (`while(region_n < new_region_n && region_n < MAX_REGION_N)` + a `pr_warn`) or raise the module's array to the monitor's max | bound = 3 lines, cannot overflow, but silently stops tracking regions ≥64 (later `mmap` of them fails); raise = tracks everything, +costs 32×`RegionInfo` in the module | the hole fix made this reachable; any boot with >8 SQLite-class domains can now walk past 64 | QEMU positive control after item 3; then it is safe on the board | **raise to the monitor's max AND bound** (belt and braces); ◆ the lead confirms |
| 10 ◆ | kernel unification: both are Linux 6.1 (FPGA: buildroot 6.1.26 + `fpgakernel.config`, 103 lines; QEMU: `components/linux` pin 830b3c6 + `kernel.config`) | one kernel source + one config with two fragments | one less pin; the QEMU kernel's capstone patches reach the board | the board kernel is the one that boots silicon; a config merge is a boot-and-see change | 1 boot | **defer** — nothing in Phase A depends on it; ◆ the lead |
| ◆ | initramfs: the unified FPGA rootfs carries the QEMU line's programs (`nested/` 176 KB, seven `*.user` ~120 KB; `rootfs.cpio` 6.1 MB, `fw_payload` 17.5 MB, dominated by the kernel and the 1.6 MB SLT domain) | trim the FPGA defconfig or keep | trim saves ~0.3 MB ≈ 2 % of the JTAG upload | a second defconfig fork is exactly the divergence just removed | — | **keep**; not worth a fork |

**Status (2026-09-08, evening).** Item 3 DONE (monitor 5011c59; QEMU tier 18/18). Item 9 DONE
(buildroot 4463cba; two-armed control `tests/runtime-qemu/run-m2-region-overflow.sh`: bounded at 64
the module refuses ids ≥ 64 and the control fails, at 96 all 23 regions past 64 are tracked and it
passes; silicon confirmation = boot sw33). Item 6(a) DONE (monitor 0a97cff; worktree chain: smoke,
five probes, child-share TRAPPED, SLT identical, module check, M-2 control). Item 6(b) DONE (the
diagnostic store was already gone; the stale comment deleted, monitor cd616c0). Item 1 DONE in
source (cd616c0; QEMU generated file byte-identical — the report macro is plain statements because
a `do{}while(0)` wrapper cost 176 lines of inert codegen; FPGA generated file differs only inside
`create_region` and `share_child_region`) — **boot sw33 DONE (fw_payload 1aabcfff1f7e): k800 + six rungs at oracle, the M-2 control
PASSED on silicon (45 regions, refusal RGNF at region_n 94, 22 regions past 64 tracked, dmesg clean)
— items 1 and 9 validated on silicon; and the Q-03 HOLE path fired 5 times on the board with no
fault, the first time it has been exercised on silicon (the caveat carried since sw30 is closed). The QEMU probes merged (capstone-qemu b9f6b00). Item 8 **DONE**: `rdtime`
permanent; the three CCSRRW-adjacent `fence.i` kept (load-bearing by R-26) and the UART-mint one
kept (untested); the other eleven REMOVED (monitor 4a12d8b) after three firmware-only boots — A (6
`.S` sites dropped), B (5 `cap_env_init` sites), C (both) — each CLEAN 7/7 at the oracles, then the
committed form booted CLEAN 7/7 as sw34 (`fw_payload 116e65accecc`, listed). Linked-firmware
`fence.i` count 163 → 152. Item 4 (null-blk) DONE as far as this tree can take it: the QEMU line's package carried and an
A0 loader regression repaired (create_dom_ko's relocatable S-mode path; buildroot commit of
2026-09-08 evening) — the split domain now loads and serves I/O; the remaining S-mode init fault
is pre-existing (Q-06, A/B against the pre-unification sbi.dom). `run-nullblk-all.sh` now fails on
a runner failure (it returned 0 before). Item 5 **DONE** (2026-09-08 evening): 5A (monitor 1de9e07) put the QEMU arm's globals copy in
8-byte units with one declarator per line — the generated QEMU file changed by exactly six lines
(4→3, 16→8, ldc/stc→ld/sd; R-10's shape), the FPGA file not at all; QEMU chain and tier green.
5B (monitor 4070620) collapsed the two arms into ONE create_domain built from the QEMU arm plus
the FPGA-only DBAS/DENT traces, the trap-vector `#ifdef` and the C-13 narrative, with the
pre-carve check through REPORT_REGION_OVERFLOW; QEMU generated file byte-identical to 5A's, FPGA
file differs only inside `_create_domain` (frame 672→576, the guards, the conditional carve and
slot, the RGNF check). Measured on the staged ELFs: every board domain packs a nonzero globals
offset (0x1000 ×7, SQLite 0x150000), so no board behaviour changes; the gpoff == 0 branch has no
board image and is unexercised on silicon. Boot sw35 (fw_payload 9d9061be1b26): k800, six rungs,
SLT select1 — all at the oracles, zero fault tags, DBAS/DENT printed 8/8. Nine differences gone;
no per-target conditional left inside the function. Item 2: probe built (silicon app with
delin-once + interp glue, freestanding host `rtpc`), QEMU pre-gate PASSED (0x22300000 then the
sentinel read back through the domain). Item 2 **DONE** (2026-09-08 night): boot sw36 (fw_payload
30f0ff5a33d2, monitor 4070620) ran the first transfer-annotated share on silicon — k800 control 4,
`RESULT revxfer retval=574619742` (call 1 0x22300000, call 2 0x2240005e: the domain read its
sentinel back through its own delin-once alias), SHA5/SHA6 2/2, zero fault tags, HOLE 0 on the old
arm, core alive after. 2B (monitor 5b27d01, wrapper 1c48f02, buildroot d3c2402): ONE
`MAKE_HOLE(region_id, 0x1239)` after the per-target type check; the board's `make_hole` clears the
CPMP association the removed block cleared, then retires the slot and prints HOLE/RGNN; QEMU
generated file byte-identical to item 5's, FPGA file differs only inside `_shared_region_annotated`
(the inline CPMP clear → `call make_hole`). Boot sw37 (fw_payload 5dd1c265a70d): same readings
plus exactly one `HOLE:0000000E` / `RGNN:0000000F` (the transferred arena, region 14). The
`REV_TRANSFERRED` handling is now one statement on both targets; what still differs at the site is
the type-check REPORT (UART tags vs `C_PRINT`), the reporting layer. The 5A nightly tier
(`/tmp/capstone-5a/nightly-20260908_194146`) finished 17/18: `beebs` FAIL(1) on `sglib-hashtable`,
whose guest printed nothing within 20 s of launch — before the loader's "Ok, good file." line, so
before any monitor code — in a tier that also logged five boot-to-login infra retries (huffbench,
minver, newlib-sqrt, sglib-arraysort, tarai ×2; the two earlier tiers today had one each) while
another user's MySQL/Bazel jobs loaded the machine. Rerun alone, first in a fresh boot, three
times on the rebuilt images (fw_jump c1a450ac5d06, sbi.dom f48906bf25a4, generated file
byte-identical to 5A): 3/3 at the marker (retval 0xc171c0de, `Globals offset = 0x0` — the QEMU
BEEBS images DO take the gpoff == 0 branch, so that branch is exercised under QEMU, only not on
silicon). Classified as an infra flake; the other 17 suites passed on the same images. The probe pair (`rtpc`, `revxfer.dom`, entry VA
0xA0000) stays staged; the SLT trio and the M-2 pair sit in `/tmp/capstone/overlay-attic` until a
boot needs them (the preflight refuses unused overlay files above the budget).

**Order that costs the fewest boots:** 3 (QEMU only) → 9 (QEMU control, then it protects every later boot) → 6 rounding (QEMU) → **one board boot** carrying 1 + 2 + 6-magic-removal + 8's firmware-only arm (batched, control first, one unknown last per the board rule — so more likely two boots: {1, 6, 8} then {2}) → 4 null-blk → 5 last with its own ladder boot. Items 10 and the initramfs need no work unless the lead wants them.

