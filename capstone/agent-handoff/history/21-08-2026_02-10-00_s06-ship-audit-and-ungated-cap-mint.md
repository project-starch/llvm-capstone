# The S-06 ship audit: one new forge path, and four reasons the squash cannot go as planned

**Date:** 2026-08-21
**Trigger:** preparing to squash `origin/fpga-testing-dev..f231b5af0` (31 commits) onto the
shared `fpga-testing-dev` branch. An adversarial audit was run first. It refuted the framing.

## 1. NEW FINDING, and it is not S-06: five non-spec instructions mint capabilities from integers

The egress rule that decides a result's capability tag is **content-derived**:

```systemverilog
core/ex_stage.sv:1174   result_metadata : {(capstone_flu_res.cap_result.metadata.cap_type != NOT_CAP), compress_cap(...)},
core/ex_stage.sv:1186   result_metadata : {(capstone_dyn_res.cap_result.metadata.cap_type != NOT_CAP), compress_cap(...)},
```

So anything that sets a non-`NOT_CAP` type mints tag = 1. Four FLU ops do exactly that with **no
`NOT_CAP` guard on the operand** — contrast every other Capstone op, e.g.
`capstone_dyn_unit.anvil:26`, which raises on a non-capability:

```
core/anvil_build/capstone_flu_unit.anvil:333  func CAPCREATE(data){ ... modify_cap_type(rd_in, CAP_TYPE_UNINIT)
                                       :342  func CAPTYPE(data){   ... modify_cap_type(rs2, rs1.cursor[0+:3])
                                       :358  func CAPPERM(data){   ... modify_cap_perm(rs2, rs1.cursor[0+:3])
                                       :366  func CAPBOUND(data){  ... modify_cap_start(rd_in, rs1.cursor)
                                                                   ... modify_cap_end(rd_out, rs2.cursor)
```

**They are not in the specification.** Checked against a live matcher — `SEAL`, `MREV` and
`REVOKE` are all found in `capstone-spec/parts/`, while `CAPCREATE`, `CAPTYPE`, `CAPPERM`,
`CAPBOUND` and `CAPNODE` return **nothing** anywhere in the spec tree.

**And nothing gates them.** `core/decoder.sv:1109` dispatches `riscv::OpcodeCustom3` straight to
`CAPSTONE_FLU`; lines 1109-1300 contain zero occurrences of `illegal_instr`, `priv_lvl_i` or
`capmode`. `capmode` appears **nowhere in `decoder.sv` at all**, and its uses elsewhere
(`commit_stage.sv:208`, `load_store_unit.sv:962`) gate the PC-capability check and one LSU path,
not this decode. The FLU dispatch (`capstone_flu_unit.anvil:511-515`) has no condition either.

**Consequence, as far as the evidence goes:** a program can execute `CAPCREATE` on a plain integer
register to obtain `cap_type = UNINIT`, at which point the egress rule sets the tag, and then use
`CAPBOUND` / `CAPPERM` to give it arbitrary bounds and permissions from ordinary integer
registers. That is construction of an arbitrary tagged capability without holding one.

**LIMITS OF THIS CLAIM, stated deliberately.** I searched `decoder.sv` and every `capmode` use in
`core/`. I have **not** exhaustively proven that no gate exists anywhere in the pipeline, and I
have **not** executed the sequence in simulation. The obvious and likely benign reading is that
these are bring-up/test scaffolding — every directed test uses them to build operands, and they
run bare-metal in M-mode — that were never intended to reach a domain. **UNRESOLVED until either
a directed test forges a capability from a domain, or the RTL author confirms `OpcodeCustom3` is
not meant to be reachable.** The directed test is cheap and is the right next step.

This is **its own issue**, not a paragraph in the S-06 report and not in the SEAL folder. One
issue, one folder, one link.

## 2. The range is not "the S-06 fix", and must not be committed as one

`origin/fpga-testing-dev..f231b5af0` is 31 commits carrying at least four separable things:

| | |
|---|---|
| `25035c4c0` | the S-06 fix |
| `9fd5507be` | **S-08** — dom-switch context width honors `metadata_en` |
| `5c5f4e3a7` + follow-ups | **S-07** — write-buffer granule co-residency |
| `scoreboard.sv` | **surviving synthesized S-07 detector**, `cap_wb_displaced_o`, self-described at `:339` as *"in the BITSTREAM deliberately"* |

The strip at `83a7d061f` cleaned `cva6.sv`/`ex_stage.sv`/`load_store_unit.sv`/`load_unit.sv` but
not `scoreboard.sv`. Committing this range under an S-06 message is the exact 2026-08-18 failure
class CLAUDE.md names.

**And the dilemma has to be stated rather than resolved silently:** extracting S-06-only content
would produce **a tree that has never been simulated and never been synthesised**. Every sweep and
the clean synthesis attach to the *full* tree. "Sim-validated" and "matches the bitstream" do not
transfer to S-06 in isolation.

## 3. What the audit CLEARED — and it strengthened the safety argument

* `git diff --stat e1140aeea f231b5af0 -- core/` is **empty**; the only difference is one sweep
  result file. The shipped RTL *is* the synthesised RTL.
* Better: `f231b5af0` is an **ancestor of `80843404c`**, which reached `write_bitstream` with
  **exit 0 and no LUTLP at all**. Since `LUTLP-1` is a bitgen DRC and not a `synth_design` check,
  "synthesised clean" only counts if the build reached bitgen — and this one did, on a superset.
  S-06 adds fan-in to `store_buffer.sv`, `pmp_data_if` and `load_unit` — all inside the very cone
  S-10b later broke — and all three shipped in that exit-0 build. **S-06 does not close a loop.**
* `cpmp_tag_i[15:0]` holds: reset `'0` (`csr_regfile.sv:3004`), two writers both carrying real
  tags (`:2414`, `:1933`), unmangled plumbing through six files, all four check sites gated, and
  the failure direction is deny-not-grant.
* `corev_apu/fpga/scripts/run.tcl` +28 is **comment-only**; the one functional line is unchanged
  context.
* The S-07 instrument in `cva6.sv` is **NOT droppable** — `run_sqlite_stages_fpga.py` has 67
  `s07` references, `decode_s07_verdict` at `:423` called from `:745` and `:1432`, reading mux
  switches 204/205-209/208/212/220.

## 4. The evidence is thinner than "sim-validated" implies

Of ten S-06 tests, **two have a proven-firing negative control** (`s06-lowhalf-zero` and
`-swap`, both `FAIL` pre-fix → `SUCCESS` post-fix in the committed sweeps). The rest:

* `fixup-tag-survival` and `cap-roundtrip-bounds` — **identical trace hash** before and after, so
  they are insensitive to the change. Fine as non-regression checks, not evidence for the fix.
* `untagged-ldc-stc-128` passed on both sides, and the test itself was modified inside the squash,
  so even a flip would not have been attributable.
* The three `s06sec-*` tests have **no pre-fix run in the tree at all**. The commit message claims
  each has a control shown to fire; that control is not in the committed artifacts, so the
  hardware side cannot see it.
* `s06sec-amo-no-resurrect` is **not in the testlist** and appears in no sweep — so the AMO/I4
  residual is an assertion, not a committed result.

**Cheap fix:** one pre-fix sweep of the three `s06sec-*` tests at `013e162fd` converts three
unproven tests into three controls.

## 5. Two instrument problems worth more than the tests they support

**`call-hot`'s trace hash flip-flopped across monotone RTL additions with no revert:**
`311aad869d14faac` at baseline and p3, `dbdb8f85417ea712` at p4 and p6audit, back to
`311aad869d14faac` at s07-fix and s07-strip — identical cycle count and exception count
throughout. Either the trace is non-deterministic for timeout tests, or another sweep was
contaminated. The three validity checks added in `f231b5af0` (duplicate names, row count,
empty-string sha) would not catch this. **A gate with an unexplained flipping row should not be
described as "byte-identical".**

**`core/tracer.sv` P6 changed what board traces MEAN, and has zero simulation coverage.** It is
the Capstone UART trace buffer (`cva6.sv:960`), not the RVFI tracer the sweep hashes — so the A/B
is *not* invalidated, and that is precisely the problem: the sweep is structurally blind to it.
The semantics changed for exactly the values S-06 is about:

```systemverilog
before:  payload = {63'b0, ...result_metadata[30:28] != NotCapType};
after:   payload = {63'b0, ...result_metadata[CapTagBit]};
```

For an untagged LDC of raw data aliasing a cap type, the old rule reported 1 and the new reports
0. **Board traces captured before and after P6 are different instruments** and must not be
compared. This belongs in the commit message and the handover.

## 6. One inference that has been reading as a measurement

"`f231b5af0` is byte-identical to what is on the board" rests on a **deduction, not a checksum**:
`ref/RATE-RULE.md:1148-1160` narrows `caplifive_s07fix.bit` to `618f4ce36` or `f231b5af0` from a
debug-mux read of `sw 237 = 0x00`, then disambiguates. Both candidates contain S-06, so the
conclusion survives — but it is inference from an instrument, and should be worded that way. The
SQLite support is also **N=3**; `RATE-RULE.md:1961` says so itself.

## What blocks the squash

1. **Message/scope.** Restructure into separate commits, or write one message naming S-06 **and**
   S-07 **and** S-08 **and** the surviving `scoreboard.sv` detector. Not "the S-06 fix".
2. **The subject line of `25035c4c0` is false as written** — "no path from raw data to a tag" is
   contradicted by its own body (the AMO/I4 residual) and now by §1.
3. **§1 needs a yes/no** on whether `OpcodeCustom3` is reachable from a domain.
4. **One pre-fix sweep** for the three `s06sec-*` tests.
5. **Re-run `call-hot` twice** at `f231b5af0` and either confirm the hash is stable or say it is not.
6. **Message must record** the tracer semantics change.

Not blocking: the `cpmp_tag` plumbing, the synthesis risk, and the `pc_metadata` content rule —
all attacked and all held.
