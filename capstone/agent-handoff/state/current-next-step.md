# Current recommended next step

## 2026-07-31 (late) — READ THE CORRECTIONS FIRST. Several instruments were misread.

### Board status: UNUSABLE as of ~21:30

Six consecutive sessions failed before running anything. The board boots (kernel reaches
~82 s) but never reaches `login:`, while emitting hundreds of
`remote fence extension is not available in SBI v1.0`. Earlier failures showed repeating
`0xDEAD…` on the UART after a wedge. Power-cycling from the driver does not clear it.
**Needs a manual reset or bitstream reload before any further board work.**

### THE CORRECTIONS — these invalidate a lot of earlier reasoning

Verified against source, not argued:

1. **`ex_commit.valid` is the EXCEPTION-valid bit, not a retirement bit.**
   `cva6.sv:500` — `exception_t ex_commit; // exception from commit stage`, wired to
   `.exception_o` at `:1800`. **Nothing in this campaign ever measured that the core stopped
   retiring.** The bit that does report it, `commit_instr_id_commit[0].valid`, lives in bank
   `debug_byte_sel = 3'b110, reg_sel = 0` (`cva6.sv:1200`) and has **never been sampled**.

2. **`stall_issue = 1` is not evidence of a hang.** `issue_read_operands.sv:390` —
   `stall_issue_o = stall_raw[0]`, asserted by any unforwardable RAW dependency. `strlen`'s
   loop is four mutually dependent instructions, so `stall_issue = 1` is its **steady state
   while running normally**.

3. **Domains run with `mtvec = ctvec = 0`, so an in-domain fault has NO HANDLER.**
   `sbi_capstone.c` zeroes all `dom_seal[]` slots and writes only 0, 2 and 3. Slot 1 *is*
   `{ctvec, mtvec}` (`csr_regfile.sv:399`) and *is* swapped on domain entry.
   **Consequence: "nothing was printed, therefore no trap was taken" — used repeatedly in
   this campaign to kill hypotheses — was NEVER a valid inference.** In 57 board sessions no
   in-domain fault has ever printed `EXCX`/`MCAU`/`MEPC`.
   NOTE: **no monitor implementation anywhere writes slot 1** — not the reference, not either
   fork. This is upstream design, not a local regression. Choosing an `mtvec` value is a
   design decision, not a patch to apply unilaterally.

4. **The load-syncer arming leak is refuted by our own capture.** It requires `req_set == 1`
   to persist; `board-regs.log:784` decoded and printed `load_syncer_req = 0` and
   `store_syncer_req = 0` on the wedged core. The `:306` vs `STC:369-370` asymmetry is still
   a real one-line difference worth reporting, but it is **not** this failure.

5. **R-14 double-counted its evidence.** The register capture attributed to an independent
   "20-line synthetic" is `sqlite_silicon.dom` built as **stage 18** — a SQLite staged build.

6. **`0x81f3c71c` was manufactured.** It was composed from eight pc bytes read ~1.2 s apart
   on a possibly-running core. Never compose a pc from separately-timed byte reads.

**Net: the "core deadlock" framing is unsupported. The signature fits a LIVELOCK IN DOMAIN
CODE equally well, and no experiment so far distinguishes them.**

### What is genuinely fixed and board-confirmed

* **Unaligned initialiser copy** (the one real root cause found today). The glue copied
  globals with 8-byte `ld` from `blob+blob_off`, and `blob_off` is not 8-aligned for 67 of
  176 globals; CVA6 does not service unaligned `ld`, QEMU does. Byte-survival `0xF0 → 0xFF`.
* `movc` destroying linear sources in `strlen`/`strcmp`/`strcpy` — indexed forms
  (`BEEBS_STRING_LINEAR_SAFE`).
* Entry, return, `sqlite3_config(HEAP)`, `MutexInit`, `MallocInit` (memsys5),
  `PcacheInitialize` all proven working on silicon (stages 0/1/7/8/9 return rc=0).

### Reverted as harmful

The granule-alignment block. It aligned the base but not the length (idx 170 carve length
262,384, `% 512 = 240`) and left **240 bytes of the memsys5 arena uninitialised** that were
zeroed before. Off by default; `INTERP_GRANULE_ALIGN=1` to re-measure. A correct version must
align the LENGTH, not the base.

### The next experiment, ready to run when the board is back

`CAPSTONE_SQLITE_STAGE=51` — the **watchdog**. Bounds the loop so a livelock RETURNS a marker
instead of spinning. It is the only experiment that yields information in all three outcomes:

* `rc = 0xB1` → the `strlen` walk never terminates ⇒ **livelock**, localised to one loop
* `rc = 16`  → completes when bounded ⇒ the wedge is a budget effect, re-base the bisection
* WEDGE      → the core genuinely stops ⇒ retires the whole livelock family

Run `wk9` (or any known-good stage) first as a health control so a board problem is not
misread as a result.

Also ready but **NOT trustworthy yet**: `CAPSTONE_SQLITE_STAGE=50` (C-14 in isolation, two
`movc` from one live scalar source). The expected back-to-back same-source `movc` pair is not
visible in the disassembly. Do not run it for a result until it is. (`movc` funct7 is `0x0A`,
not `0x14` — the top byte `0x14` is `funct7 << 1`.)

### Traps that bit repeatedly today — read before touching the board

1. **Prune `build/target/` AND the overlay, but KEEP `sqlite_silicon.dom` and
   `sqlite_host.user`.** Buildroot copies the overlay into `target/` and never deletes, so
   the initramfs regrows; at 26–46 MB domain loading silently breaks while the freshness gate
   still passes. But those two files are the gate's reference artifacts — prune them and it
   correctly refuses to flash.
2. **Verify the artifact changed** (hash, store count, section size) before believing any
   result. Three probes tested nothing today: a pointer ternary hit the i128 `SELECT_CC` gap;
   uninitialised `static` arrays produced zero cap-init leaves; and declaring all five arrays
   in one TU put every array in every build.
3. **Ad-hoc board scripts must be backgrounded or install a signal handler** — a foreground
   script killed at the tool's 2-minute limit skips its `finally` and leaves the board locked.
4. A **ladder rung is not a control for this glue**: `build-ladder-domain.sh:22` defaults
   `DOMAIN_GLUE=generated`, so it uses `start-gp-captable-generic.S` instead.
