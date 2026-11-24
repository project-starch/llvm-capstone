# b5b2-bare-2026-11-24

B5b-2 on silicon: the B5b-2 directed tests as bare JTAG images, for B5b-2's full build (capstone-ariane b5b2-step6
0c0458149: steps 1-6 batched with CPTAG whole, plus step 7's test). Design, rulings and simulation readings:
llvm-capstone capstone/docs/plans/2026-11-24-b5b2-trap-route.md.

**Siblings:** `csmint-bare-2026-11-23` is B5b-1's kit. Its `sublet-life` is the control here too, rebuilt with this
kit's id character. B5b-2 runs bare kits only: CAPENTER is illegal under the Sublet format, and the current monitor
executes CAPENTER at init.

**Images** (the tests unchanged from capstone-ariane 0c0458149 `verif/tests/custom/capstone/`; `-D` as in
testlist_b5b2.yaml):
- `sublet-life`: the known-good control, first in the boot (B5b-1's Sublet lifetime arms on the physical pool).
- `b5b2-illegal`: step 1's positive control, every capability instruction the reference deleted, then the trap
  count. `b5b2-illegal-runv` (`-DARM_RUNV`): csrunv's arm, run alone.
- `b5b2-uprot`: step 2, the LSU's admission gate by privilege (protected U, S rule 1, M plain). `b5b2-uprot-mis`
  (`-DUP_MIS`): four more misaligned protected-U arms.
- `b5b2-rule2`: step 3, rule 2 in the MMU (a plain S access reaches no user page while srevroot != 0).
  `b5b2-rule2-pair` (`-DR2_PAIR`): the leak test's phase E.
- `b5b2-sref`: step 2b, supervisor references (LDC/STC of S and M through a plain base keep the tag; LCC 8 in S).
  `b5b2-sref-gen` (`-DSR_GEN`): LCC 8 at a non-zero generation.
- `b5b2-scratch`: step 4, sscratch and mscratch hold a reference with its tag. `b5b2-mwrite`: an srevroot write keeps
  only the aligned bits (route.S masked_write).
- `b5b2-raw`: step 5, CLDRAW and CSTRAW. `b5b2-rawlcc`: a capability instruction reading a raw register as rs1 leaves
  its raw_tag (fix 2).
- `b5b2-cptag`: step 6 part 1, CPTAG's ordering and the page bitmap. `b5b2-cptag2`: part 2, the acknowledge (the exact
  OR; an empty page's bit cleared).
- `b5b2-nsedge`: step 7's namespace must-pass (srevroot switched between two accesses).
Most of the tests build their capabilities with the capcreate family, which stays legal in this hash.

**The wrapper** is `csmint-bare-2026-11-23`'s (= `r79-pmp-bare-2026-11-23`'s), with three kit-local changes, all in
`inc/`:
- `board_wrap.S` no longer includes the board header ahead of the test. Each test includes `asm_insn.h` itself, which
  resolves to `inc/asm_insn.h`, so the macros are defined from the same line as in the tree. Included first, they
  expanded macro names in b5b2-sref.S's leading `#` comments. sublet-life's image is byte-identical either way.
- `RVTEST_PASS` is the board report (`CAP_PASS`). The B5b-2 tests end in RVTEST_PASS, a tohost store, which prints
  nothing on the board: the vector reaches the UART only through CAP_PASS.
- `BOARD_REC_PLAIN`: a recorder for b5b2-sref, which names every GPR but tp. It uses plain M stores, x4, and spills
  rs1, x5 and x6. `out/xcheck/b5b2-illegal-plain` is the same test as `b5b2-illegal` with this recorder. It is
  simulation only, never booted, and must read the same vector.

`build.sh` builds with riscv64-unknown-elf-gcc, as the tree builds these tests. Clang expands macro names inside `#`
comment lines. Recorder registers per test come from pick-regs.py (`satpstale-bare-2026-10-19`).
`images/SHA256SUMS` pins the images. `PREREG.md` holds the vectors, simulated from THESE images, and the grading rule.
Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG`, and `SUP_REC_ADDR` from each `.rec`).

**Silicon state between images:** the runner power-cycles before each image.
- A reset clears the page bitmap (wt_axi_adapter.sv tbm_clr_q), and TAG_FILL zeroes a page's tag bytes before its
  first tag write. So no stale DRAM tag byte is read, and CPTAG starts from unmarked pages.
- All test data is PROGBITS (`.zero` in `.data`), so the images carry their zeros. The prologue rewrites `.data` and
  zeroes `.bss`.
