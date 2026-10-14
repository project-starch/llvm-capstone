# Q-04 decided: MOVC copies untagged data and moves only capabilities that may not be duplicated

*2026-10-14. On that day the project lead decided Q-04 ("Option 1 übernehmen, wie vorgeschlagen,
vorsichtig sein": adopt option 1 as proposed, and be careful). This file records the decision and
the audit that came before any change. It also lists what changed in the spec, QEMU, the compiler
and the SDK, what the RTL change must be, and the evidence. The question and the earlier
recommendation are in
[`2026-09-24-q04-movc-integer-source.md`](2026-09-24-q04-movc-integer-source.md).*

## The rule

With `rd != rs1`, MOVC writes `x[rs1]` to `x[rd]`, contents and tag. It writes `cnull` to
`x[rs1]` only if `x[rs1]` is a **tagged** capability whose type is not NONLIN. Untagged data, null
included, and tagged NONLIN capabilities stay unchanged. `rd == rs1` is a no-op. MOVC raises no
exception.

This is **not** "consume only LINEAR". REVOKE, UNINIT, SEALED, SEALED-RETURN and (RTL only) EXIT
are still consumed.

### Semantics matrix

| `x[rs1]` before | `rd != rs1`: `x[rd]` after | `rd != rs1`: `x[rs1]` after | `rd == rs1` |
|---|---|---|---|
| untagged 0 (cnull) | untagged 0 | unchanged (untagged 0) | no-op |
| untagged, any other 128-bit pattern (including a capability's bits) | the same bits, untagged | **unchanged** (was: cnull on the RTL) | no-op |
| tagged NONLIN (type 1), live or stale | the same capability | unchanged | no-op |
| tagged LIN (0) | the same capability | cnull | no-op (stays LIN) |
| tagged REVOKE (2) | the same capability | cnull | no-op |
| tagged UNINIT (3) | the same capability | cnull | no-op |
| tagged SEALED (4) | the same capability | cnull | no-op |
| tagged SEALED-RETURN (5) | the same capability | cnull | no-op |
| tagged EXIT (RTL `CAP_TYPE_EXIT`, no QEMU or spec type) | the same capability | cnull | no-op |
| `rs1 = x0` | cnull | x0 (always 0) | — |
| `rd = x0` | discarded: x0 still reads 0 and stores cnull | follows the row of `x[rs1]` (a LIN source is destroyed) | — |

Only the second row changes. Every tagged row is the same as before on every implementation.

The spec defines an integer as XLEN bits. An untagged register image wider than that (the second
row's upper half) is QEMU's and the RTL's register state (QEMU `scalar_hi`; the RTL's NOT_CAP register
contents), not a spec-level contract. Both implementations keep it, and the QEMU suite pins QEMU's.

## Security argument

- **An untagged value carries no authority.** Neither register gains a tag, permissions or a live
  revocation identity from the copy. Any integer can be made with `li`. Copying a capability's
  *bits* in an untagged register does not create authority either: the copy faults 24 on use
  (QEMU case `untagged_capability_bits`).
- **Linearity is unchanged.** Every tagged type that the old rule consumed is still consumed. The
  only exemption is keyed on the tag, not on the type, so a stale or sealed capability cannot use
  it.
- **Revocation is unchanged.** A stale tagged NONLIN capability was already copyable and stays
  copyable; no copy becomes live. The virtual collector demotes stale registers to untagged
  cursors before it recycles identities (`capstone_supervisor.c`, `gc_register` and
  `virtual_collect`). After that, MOVC keeps both untagged copies, and neither regains authority
  when the identity is reused (QEMU cases `stale_copy_before_collection` and
  `stale_copy_after_collection_and_reuse`). The collector must still sweep every tag before an ID
  is reused, as before.
- **It matches STC.** STC already keeps an untagged or NONLIN source, in the spec's
  capstone-bootstrap line (`mem-access-insn.adoc:105`) and in the RTL (`capstone_dyn_unit.anvil:525` and `:544` at the dev pin 776d9d85).
  Register-to-register and register-to-memory copies now follow one rule.

## Audit: does anything rely on MOVC zeroing an integer source?

**No.** What was searched and found:

| Where | MOVC uses | Relies on zeroing? |
|---|---|---|
| compiler (`llvm/lib/Target/Capstone`) | `copyPhysReg` emits MOVC for every GPCR copy; `eliminateFrameIndex` emits MOVC for a zero-offset sp/fp copy; `PseudoCapGlobalBase` and the direct-call target are fused to avoid a LINEAR MOVC source (C-46) | No. MOVC is modelled as a pure copy (C-46: `$rs1` is a plain use), so compiled code cannot depend on the source being zeroed. `CapstoneDeadRegisterDefinitions` only renames GPR defs to x0, never a GPCR def, so backend-generated copies never use `movc zero, rs` (inline or hand-written assembly can). |
| `ports/musl-capstone/runtime/start-musl.S` | `:98` `movc sp, t5` (t5 from SPLIT); `:236` `movc t1, ra`; `:454` `movc a1, a2` | No. `:236` clears ra with `li ra, 0` on the next line. The scrub path (`.Lmusl_scrub_return`) clears every register with `li`, never with MOVC. |
| `my_first_domain/start.S:216` | `movc t1, ra` | No; `li ra, 0` follows. |
| musl `altstack.S`, `set_thread_area.S`, `pthread_arch.h`, `delegate.c` | stack switch; `movc tp, a0` then `li a0, 0`; `movc %0, tp` | No. |
| runtime (`virtual/contract.c`, `pthread-contract.c`, `tests/application/bounded-stack.S`) | `movc %0, tp`, `movc sp, t0` | No. |
| kernel module `capstone_vm.c` | none (funct7 0x0a does not occur) | — |
| capstone-c (`capstone-c-mapping`, 8cda52c = dev pin) | `move_reg` emits MOVC for 16-byte values; the REVOKE, DELIN and DOMRETURN lowering | No. Its scrub is `li reg, 0` (`codegen.rs:781`). |
| monitor, capstone-sbi (`capstone-sbi-threads`, 4674ab6) | `sbi_capstone.S:52` `MOVC(a0, sp)`; `:197` `MOVC(sp, a0)`, followed by `li a0, 0; li a1, 0`; `sbi_capstone.c:2843` `movc(a0, %0); j resume_smode` | No; every source is a capability, and the one live afterwards is cleared with `li`. |
| caplifive-opensbi (`caplifive-opensbi-threads`, cf344cf) `sbi_capstone_init.S:58` | `MOVC(a2, a3)` of a SPLIT result | No. |
| monitor compiled code | the 2026-09-24 exposure run saw one integer MOVC per boot (pc `0x80020a50`, x9 = 1, `split_out_cap` → `read_cpmp`) | No; under the RTL rule it was zeroed without effect, and compiled code assumes a copy. |
| tests | `rtl-smoke` sup-* tests move tagged SEALEDRET/NONLIN values only; `silicon-ladder/s12*` use `movc t, zero` (x0 source, unchanged); `silicon-ladder/movcprobe_kernel.h` returns 49 when the source survives (QEMU, and the board after Q-04 b) and 7 on today's bitstream; `rtl-smoke/live-source-copy` runs the STC/LDC pair the compiler emits for integer, NONLIN and LIN sources (it never executes MOVC) | Only the two probes that measure the RTL difference on purpose. |

**Prior art** (`git log --all --grep 'Q-04\|C-32\|movc'`):
- `2026-09-24-q04-movc-integer-source.md` (6ac8b149) recommended exactly this rule, with the same
  RTL diff.
- #96 merged the decision documents as proposals.
- `CAPSTONE_MOVC_NULL_SCALAR` (capstone-qemu#7) and its exposure run found `iconv_open` as the
  only corpus verdict that changes.
- C-32's live-source copy rule (da04d49e, #119) is the compiler workaround this decision lets the
  virtual profile drop.
- The compiler-only narrowings `compiler/no-loop-capability-spills(-stable)` were dropped: they
  keep 294 of 297 pairs, because R-34 leaves scalar accesses unchecked.

## What changes where

| Where | Change | Branch |
|---|---|---|
| record | The `movc` row of `CapstoneISASemantics.md` holds the rule, per profile, next to the untagged CINCOFFSET/SCC rule. The academic-spec text is not updated (no write access, and it already differs from the implementations elsewhere); spec texts for `master` and `capstone-bootstrap` are kept locally as `spec/movc-untagged-source(-bootstrap)`. | this branch |
| QEMU | The rule on the virtual profile (`x-capstone-u-mode`), which already kept an untagged source. The physical profile now nulls an untagged source, as the RTL still does, because physical QEMU models the silicon: the same split the maintainer gave the untagged CINCOFFSET/SCC rule (capstone-qemu 2794082073). **One deviation fixed:** `movc zero, rs` wrote `rs` into x0's storage, so a later `stc zero` stored that value instead of cnull. New suite `tests/movc-untagged-source` (67 checks; `--legacy` 85). | capstone-qemu `isa/movc-untagged-source` |
| compiler | No source change. A lit test shows that the feature, when it arrives as a function attribute (the SDK's path), removes the STC+LDC rewrite. | llvm-capstone `isa/movc-untagged-source` |
| SDK (virtual profile) | `-Xclang -target-feature -Xclang +movc-keeps-integer-source` in the application, runtime, builtins and musl codegen. The physical profile and the compiler default are unchanged: the bitstream still nulls. | same |
| probe | `movc-rule` turns the feature off explicitly, so it still tests the rewrite under the new SDK default. | same |
| RTL | Q-04 (b), below. Not applied. | — |

### Q-04 (b): the RTL change (not applied)

`capstone-ariane` `core/anvil_build/capstone_flu_unit.anvil`, `func MOVC`, the preserve branch at
line 14 (same text at the dev pin `776d9d85` and at `f6ec6c198`):

```diff
-            if(data.cap_rs1.metadata.cap_type==cap_type_t::CAP_TYPE_NONLIN){
+            if((data.cap_rs1.metadata.cap_type==cap_type_t::CAP_TYPE_NONLIN)||
+               (data.cap_rs1.metadata.cap_type==cap_type_t::NOT_CAP)){
```

The branch body already returns `rs1` and `rd` unchanged, so there is no new state, signal or
stage. CAP_TYPE_EXIT stays consumed. The cost is one more comparison and an OR on the 3-bit type
field that the branch already reads.

Before it ships:
1. **Lint** at the baseline (`ANVIL_UNOPTFLAT 0`), and read the generated SystemVerilog for the
   new comparison.
2. **Directed simulation**, one case each:
   - NOT_CAP non-zero and NOT_CAP zero, with `rd != rs1`: both registers hold the value afterwards;
   - NOT_CAP with `rd == rs1`;
   - each of LIN, REVOKE, UNINIT, SEALED, SEALEDRET and EXIT: consumed;
   - NONLIN: kept;
   - `rd = x0` with each kind of source: x0 stays zero;
   - `rs1 = x0`;
   - the very next instruction reading `rs1` and then `rd`, to check forwarding and the dual
     write-back of the FLU result pack.
3. **Regression.** Rerun `tests/rtl-smoke/live-source-copy/`, which runs the STC/LDC slot pair for
   integer, NONLIN and LIN sources and never MOVC. Comparing the pair with MOVC for every type is
   part of the directed tests in step 2.
4. **Synthesis.** Report WNS and LUTs against the same tree without the change. The prediction is
   the R-30/R-31 pattern: timing and area essentially unchanged and a different bitstream hash.
   Do not quote timing or area until synthesis has run.
5. **Board.**
   - `silicon-ladder/movcprobe_kernel.h` must read 49 (today: 7).
   - The SQLite stage-50 probe must read 55.
   - The Sublet -O2 image's `--stats` must show non-zero lookaside slots (today: 0; the emulator
     shows 25,010).

Until then, every physical-profile image keeps the C-32 rewrite (the compiler default).

## What does not change

- **R-34** (the LSU exceptions lost in the request cycle, and the M-mode gate on the scalar
  capability check) is separate. MOVC keeping or zeroing its source mitigates neither. A zeroed
  base is still not a guaranteed fault on today's RTL, which is why a compiler-only narrowing of
  C-32 cannot be sound.
- **C-46** stays open. LLVM still models MOVC as a pure copy, which is wrong for a LINEAR source.
  It does not say which sources are consumed, and the feature changes nothing about linear
  sources. Under the old rewrite, STC nulls a LINEAR source and the first LDC empties the slot.
  Under the feature, MOVC nulls the source. Both are a move.
- The compiler default (`-movc-keeps-integer-source` off) and the physical SDK.
- Historical board results stay identified by the MOVC semantics and compiler they ran with.

## Evidence

All measured on 2026-10-14 by this lane. Hosts and binaries are as named; p12 is shared, so its
times are not timings.

**QEMU** (p12; capstone-qemu `isa/movc-untagged-source` on `virtual-capstone` 70e98dd390, which
has #22, #23, #25 and #26):
- `tests/movc-untagged-source/run.sh`: 67/67.
  - 45 virtual and M1 checks under the rule.
  - 15 physical builds under the RTL's rule, plus 5 physical default builds of the differing cases,
    which are correctly rejected.
  - 2 harness controls.
- `--legacy` (`CAPSTONE_MOVC_NULL_SCALAR=1`, every context under the RTL's rule): 85/85.
- Before the x0 fix (91b96f2b), `x0_destination` and `consuming_types_self_and_x0` failed in every
  context.
- Mutants of `helper_csmovc`, each built from its own source tree (failing checks out of 67):
  - no physical branch: 10, exactly the physical differing cases;
  - "consume only LIN": 15 (REVOKE, UNINIT, SEALED, SEALED-RETURN, and the self and x0 moves);
  - "consume nothing": 27;
  - "consume NONLIN too": 17;
  - "write x0": 8.
- Other suites on the same binary: `virtual-capstone-runtime` 69/69, `store-tags` 4/4,
  `exact-bounds` 4/4, `virtual-capstone-m1` 69/69, `capstone-untagged-arith` 6/6.

**Compiler** (p12, `~/cc-ab/build-C80b`, assertions ON):
- The new lit test passes, and both of its mutations are rejected.
- `llvm/test/CodeGen/Capstone` at this branch: 120/121. The one failure is the known
  `shared-patches-present.test`.

**SDK** (p12, compiler `~/cc-ab/build-C80`). Two SDKs were built from copies of the trees behind the
existing baselines, each changed only by this branch's flag edit: `t5k` = tree5 (whose sdk5 built
the measured xz) and `Dk` = the gate tree `~/hosted-ab/D/tree` (916800b34d9b).
- The flag reaches sdk.json and 79 runtime compile lines, with no "not a recognized feature"
  warning.

**557.xz_r**, rebuilt from SPEC's own compile lines (`make.out`) with sdk5 and with t5k:
- The sdk5 replay is byte-identical to the measured `~/spec-cap/mix/xz-cap.dom`.
- Adjacent STC+LDC frame-slot pairs (sp or s0) in the whole linked binary: 1548 with sdk5, **0**
  with the feature. In loops in xz's own `-S` output: 585 against 0. My counter takes a loop to be
  a label up to a later backward branch to it. The earlier figure of 297 came from a different
  counter that I could not locate.
- `.dom`: 470,896 bytes against 464,672.
- The output oracle passes for both builds ("Uncompressed data compared correctly", "Tested 4 MiB
  buffer: OK!", exit 0): once on p12, and in every p13 run below.

**Timing** (p13, `taskset -c 2,42`, one boot per run, interleaved base/keep/keep/base/base/keep, the
same QEMU binary as above, p12-built, sha256 d39be61a…). Load average 7.1-8.1 from other agents on
socket 0, so the timings are approximate.

| | base (sdk5) | keep (sdk5's tree + flag) |
|---|---|---|
| xz guest time, s (`/proc/uptime` around the run) | 77.75, 77.84, 77.83 (mean 77.81) | 71.33, 71.06, 71.41 (mean 71.27, **-8.4 %**) |
| micro `ptr_increment`, ns/iter | 545.11, 575.72, 543.19 | 30.33, 30.30, 30.27 |
| micro `call_ret`, ns/iter (control: no copy in the loop) | 427.84, 423.09, 427.88 | 426.86, 427.55, 425.73 |
| micro `empty_loop`, ns/iter | 9.97, 9.92, 9.90 | 9.90, 9.87, 9.94 |

These are emulator times. They say nothing about a board.

**MOVC probe** (`movc-null-scalar/movc_test.c`, built with t5k; the guest is on this branch's QEMU,
p12):
- Switch off: `rule` and `keep` both print `b=5 c=5`, `got=0x5000` and `n=3`.
- Switch on: the probe prints `b=5 c=0` and the notice (the monitor's integer MOVC at
  `0x80020a50`, as on 2026-09-24). `rule` prints `got=0x5000` and `n=3`. `keep` prints `got=0x1`
  and `n=1`, the positive control.
- After Codex's review, the runner rejects that `rule` reading: its runtime and libc were built for
  Q-04 (see the probe commit).

**Application gates** (p12, `run-gates-p.sh`). `~/hosted-ab/platform-movc` links the usual adapter
and images and this branch's QEMU. The script's SDK, ports and tree arguments were the Dk SDK, ports
built from the Dk tree (memcached, sqlite, mruby and perl; each port's own SDK is built from that
tree, so it has the flag) and the Dk tree. That is how the gates used this SDK and not
`~/hosted-ab/D/sdk-out/sdk`.
- Pass 1 (`gates/movc`): hosted rc=0; nodes rc=0; malloc rc=0; memcached rc=1, with 104 PASS and
  only `fx24_trigger` failing (the known timing flake). Its runtime gate never ran: rc=1 after 1 s,
  because the mruby port's build had timed out downloading musl.
- mruby was rebuilt with the tarball seeded, and pass 2 (`gates/movc2`) ran: hosted rc=0; runtime
  rc=0 (sqlite, mruby and perl images all exit 0, `VIRTUAL_RUNTIME_DONE`).
- Every gate thus has a result with this SDK, and all match the expected pattern.

## Review

Codex (gpt-6.1-sol, xhigh) reviewed the memo, the audit, the QEMU fix and tests, the lit test, the
SDK and probe changes, the spec texts (kept locally) and the RTL note.

Verdict: the core is sound. That covers:
- QEMU's consumption predicate and the x0 guard;
- the RTL predicate (NOT_CAP preserved, EXIT still consumed);
- the per-function lit test;
- the SDK's coverage of its C components and musl;
- the audit: no integer-zeroing dependency.

Findings, and what was done with each:
1. Major: the probe's `rule` variant links feature-on support code, so a legacy pass is
   meaningless. **Fixed:** the runner rejects it with that reason; self-test 49/49.
2. `--legacy` accepted any guest failure, so later legs went untested. **Fixed:** exact
   MOVC_LEGACY expectations, every leg must pass.
3. The matrix lacked self and x0 moves of REVOKE, UNINIT and SEALED, and a full-width value
   over a tagged destination. **Fixed:** case 16 and case 2.
4. Standalone CMake callers do not give the builtins the virtual flags. **Not changed:** this
   predates the branch and applies to every virtual flag. The hardfp lane's
   `VirtualCodegen.cmake` adds them to the builtins. Objects built without the feature stay
   correct.
5. The spec defines XLEN-bit integers, not 128-bit untagged images. **Fixed in this memo:** the
   wider image is labelled implementation state.
6. The memo overstated `rtl-smoke/live-source-copy`. **Fixed:** the claim is narrowed.
