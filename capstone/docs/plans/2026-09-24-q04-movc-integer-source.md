# Q-04: should MOVC consume an integer source? Recommendation: no

*2026-09-24. A decision for the project lead, with the spec's owners for the text and the hardware
side for the RTL. It supersedes nothing: item 4 of `DECISIONS-WAITING-2026-09-10.md` states the
question, and this file is the recommendation and the evidence for it.*

## The question

`MOVC rd, rs1` copies a register, and consumes the source unless the source may be duplicated. The
spec's clause is *"If `x[rs1]` is not a non-linear capability (i.e., `type != 1`), write `cnull` to
`x[rs1]`"* (`capstone-academic-spec`, `parts/cap-man-insn.adoc:37-38`). Read literally, it also
consumes an **integer** source. The RTL implements the literal reading, and QEMU does not:

| | integer source of `movc` | where |
|---|---|---|
| RTL | zeroed (cnull is `NOT_CAP` with cursor 0); no exception | `capstone_flu_unit.anvil:6-27`, `capstone_unit.anvilh:395` |
| QEMU | kept | `op_helper.c` `helper_csmovc`: nulls only `tag && type != NONLIN` |
| QEMU with `CAPSTONE_MOVC_NULL_SCALAR=1` | zeroed, as the RTL | capstone-qemu#7 |

The difference is live. C-32: the SQLite port's lookaside was silently off on the board at -O1/-O2,
while every emulator pass said it was on.

## Recommendation: (b), exempt integers. Only a capability that may not be duplicated is consumed.

Four reasons. Each is checked against its source.

1. **STC already does exactly this, in both the spec and the RTL.** Spec, `mem-access-insn.adoc:105`:
   *"If `x[rs2]` is a capability and `x[rs2].type` is not `1` (non-linear), write `cnull` to
   `x[rs2]`."* RTL, `capstone_dyn_unit.anvil:439` and `:458`: STC nulls its source only when
   `cap_type != CAP_TYPE_NONLIN && cap_type != NOT_CAP`. Today, storing an integer keeps it while
   copying it register to register zeroes it. (b) removes that inconsistency and adds no rule.
2. **The literal reading is an accident of an edit.** Before spec commit `a1db3c2` (2024-01-04, *"MOVC
   now works with non-capabilities without generating faults"*), an integer source **raised an
   exception**. That commit deleted the exception and left the consumption clause untouched. No
   version of the spec ever deliberately zeroed an integer. The section's own first line is
   *"Capabilities can be moved…"*, and both operands are annotated `(C)`.
3. **The alternative (a) cannot be implemented in the compiler.** (a) means the compiler must never
   copy an integer-valued pointer with `movc` and then read the source. But:
   - Whether a value is untagged is known only at run time. A pointer loaded from memory can hold an
     integer, and so can a merged value.
   - There is no other copy instruction. Integer instructions read a capability's cursor
     (`existing-insn.adoc:6-7`), so `mv` strips a tag, and `movc` zeroes an integer. A value that
     may be either has no one-instruction copy that keeps the source.
   - Design A (`46c53b7b6ae2`) was chosen for exactly this and measured not to remove the live site
     (C-32).
   So (a) leaves a class of silent wrong results on silicon with no known fix.
4. **A capability-carrying `uintptr_t` (intcap) needs (b).** Copying a `uintptr_t` means copying a
   value of unknown taggedness, and under (a) every such copy whose source is read again loses an
   integer (`plans/intcap-uintptr-model.md` on branch `compiler/intcap-design`, WP1).

**Security is unchanged.** The exemption is keyed on the tag alone. Linear, revocation,
uninitialised, sealed and sealed-return capabilities are all still consumed, so linearity and
revocation hold as before. An integer carries no authority: any integer can be made with `li`.

**What would change the recommendation:** software that relies on `movc` zeroing an integer source,
for example to scrub a register. The production hand-written `movc` found (`pthread_arch.h:38`,
`set_thread_area.S:24`) do not. The compiler models `movc` as a plain copy (C-46), so compiled code
cannot rely on it either. The monitor's source was not scanned; see the measurement below for what
its boot does under the RTL's rule.

## What to request

**Spec** (both `capstone-academic-spec` and `capstone-spec`, `parts/cap-man-insn.adoc`, MOVC), in
STC's wording:

```
. If `x[rs1]` is a capability and `x[rs1].type` is not `1` (non-linear), write `cnull` to `x[rs1]`.
```

**RTL** (`core/anvil_build/capstone_flu_unit.anvil:14`, MOVC with `rs1 != rd`). Use the same test STC
already uses:

```
-            if(data.cap_rs1.metadata.cap_type==cap_type_t::CAP_TYPE_NONLIN){
+            if((data.cap_rs1.metadata.cap_type==cap_type_t::CAP_TYPE_NONLIN)||
+               (data.cap_rs1.metadata.cap_type==cap_type_t::NOT_CAP)){
```

- No new signal: the field is already read there, and the write-back of `rs1` already exists (the
  NONLIN branch writes `rs1` back unchanged).
- The FLU is an Anvil unit, and the lint baseline has `ANVIL_UNOPTFLAT 0`.
- It still needs the lint gate, a directed simulation test (`movc` of `NOT_CAP` with `rs1 != rd` and
  with `rs1 == rd`), synthesis and a board pass, as any RTL change does.

**QEMU:** nothing. Its default already is (b). `CAPSTONE_MOVC_NULL_SCALAR=1` stays, as the way to
reproduce a bitstream without the change.

**Batch it.** WP1 of that plan has a second change of the same kind: `SCC` and `CINCOFFSET` on
an untagged value produce an untagged value with the new address instead of trapping. It is also the
lead's decision. If it is ruled at the same time, both go into one synthesis cycle.

**Write the predictions down before synthesis.**
- Synthesis: timing and LUT counts essentially unchanged, with a different bitstream hash. This is
  the R-30/R-31 pattern: the same structure, different LUT contents.
- Board, with the changed bitstream:
  - the SQLite port's stage-50 probe (`sqlite_capstone_domain.c:2593`) reads **55**. On the current
    bitstream it reads 50. No reading of it is recorded, so run it on both.
  - the Sublet port's -O2 image's `--stats` shows a **non-zero** lookaside count. On the current
    bitstream it is 0, against 25,010 hits on the emulator (C-32, boot sw8x-o2stats).

## Until the new bitstream is on the board

Every board result from an optimised image is exposed wherever a program copies an integer-valued
pointer with `movc` and reads the source again. `capstone/tests/runtime-qemu/movc-null-scalar/exposure.sh`
measures how much of the regression corpus that is. It runs the same programs on the same QEMU twice,
keeping and zeroing the integer. The result is below.

**Measured 2026-09-24** with `exposure.sh`:
- compiler: `dev`'s (`1a08706b6344`);
- tree: `dev` plus #89;
- QEMU: capstone-qemu `c128-qemu-merge` (`d4cec0de`) with the switch merged in.

Controls held:
- the switch test passed first (probe `b=5 c=5` against `b=5 c=0`; the C-32 shape keeps its address
  against returning its null arm);
- every one of the 251 boots with the switch on printed its notice, and none of the 252 with it off
  did.

| what | ran in both arms | differs |
|---|---|---|
| nightly, 26 suites | 22 give a QEMU verdict | none |
| BEEBS | 81 wrappers, the same 5 fail in both | none |
| hostcall-all, each probe on its own | 3 of 12 | none |
| musl file/stdio/write probes | 3 | none |
| musl libc-test, merged (chunked, and what a chunk hid re-run one per boot) | 50 built and run | **`iconv_open`: passes with the integer kept, faults with it zeroed** |
| CPython 3.13.7, the demo image (`dev`, the compiler PRs, C-46 and TLS): `checks.py` twice per boot (hash seed 0 and 1) | 6 checks × 2 | none: 6/6 both times in both arms, and the "on" boot's log carries the switch's notice |

The same `iconv_open` result came out of an earlier run of the same measurement, at the same pc.

**What the `iconv_open` fault is.** musl encodes a simple conversion descriptor as an integer:
`combine_to_from()` returns `(void *)(f<<16 | t<<1 | 1)` (`src/locale/iconv.c`). libc-test, at -O1
with `dev`'s compiler, keeps that descriptor in `s5` and passes it to `iconv` three times, each with
`movc a0, s5`. Under the RTL's rule the first copy zeroes `s5`. The second `iconv` call receives
`cd = 0`, takes its pointer path, and faults at its first load (`ldc a5, 0(a0)`, cause 24). **This is
a case (a) cannot fix in the compiler.** The integer becomes a pointer inside `iconv_open`, so the
caller only ever sees a returned pointer. No analysis in the caller can know it holds an integer,
and the planned fix for C-32's known shape would not reach it either. On today's silicon, any
program that uses one `iconv` descriptor twice is exposed.

**What this does not cover:**
- **9 of the 12 hostcall probes.** They fail in both arms, because capstone-qemu as pushed halts
  them in the monitor at `file-open-close`. This machine's shared QEMU build passes them, but it
  carries unpushed changes; #95 makes that visible in every nightly report.
- **Two suites give no verdict in either arm:** `nullblk-all` (the guest kernel oopses loading a
  module) and `sqlite-slt` (its harness does not build on this host).
- **Programs outside the corpus.** The SQLite Sublet port's lookaside (C-32 itself), FFmpeg and
  whisper were not run. Their board numbers from optimised images carry the same exposure as
  before.

**The compiler fix for C-32's known shape is deferred.** Once (b) is on the board, C-32 is a harmless
choice between `movc` and `mv`, and a fix would be thrown away. The measurement adds a reason: the
one new case it found (`iconv`) is out of any compiler fix's reach. Doing it anyway is worth it
only if the reflash is weeks away and a reported program is shown to hit the known shape.

## What this does not decide

- Nothing about which tagged types are copyable. `type != 1` stays the rule for capabilities.
- Not the SCC/CINCOFFSET change. That one is the lead's separately.
- Not the paper's framing of any earlier board number. The measurement only says which ones to check.
