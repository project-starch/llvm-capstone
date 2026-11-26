# Stage 2 step 4 on Line B: fetch bytes of CRVC, Zcb with a capability base, and both, on the trap twin (pre-registration, 2026-11-26)

The plan is plans/2026-11-25-stage2-trap-route-decoder-spec.md (Line B section, and the mode home converged with rtl).
This measures what stage 2 removes from executed instruction bytes on Line B's footing, before any RTL. The cycle
band and the go/no-go for a stage-2 seal are pre-registered from these readings afterwards, as the plan says.
Written before any measuring boot. The twin controls in "The twin" ran first, and they decide nothing here.

## Arms

All four arms use the same compiler binary, llvm-sinkland clang sha256 35580b17947b74ec, and its lld
(c45af5acebd278ec), with B5TL1's flags and SDK. Each arm's images are in perf/cclever/arms/<arm>/exes.

| arm | features beyond B5TL1's | role |
|---|---|---|
| LSHOFFL | none | the control: LSHOFF's images on 10 of 11 programs; gcc differs only by the embedded compiler revision string |
| CRVCB | +cap-crvc | every image flagged EF_CAPSTONE_CRVC (e_flags 0x101) |
| ZCBB | +zcb +cap-base-zcb | |
| CRVCZB | +cap-crvc +zcb +cap-base-zcb | the combination, never measured on P1 |

**Static text change against LSHOFFL** (sizes, not fetch bytes):

| program | CRVCB | ZCBB | CRVCZB |
|---|---:|---:|---:|
| perlbench | -9.69 % | -0.72 % | -10.38 % |
| gcc | -10.89 % | -0.45 % | -11.35 % |
| mcf | -1.63 % | -0.08 % | -1.70 % |
| lbm | -0.46 % | -0.00 % | -0.46 % |
| x264 | -5.41 % | -0.93 % | -6.34 % |
| imagick | -7.04 % | -0.31 % | -7.34 % |
| nab | -4.29 % | -0.46 % | -4.75 % |
| xz | -5.37 % | -0.33 % | -5.70 % |

## The twin

- **The QEMU binary** is capstone-qemu compiler/stage2-crvc-trap c42ff84d0799 (the decode at d69a81502d; route.S's CRVC read-back case on top, 29/29), sha256 457477d73dde1a2d. It is the
  trap twin qemu-crv (3a9fc39167, sha256 ea5c05f31d93feb2) plus the CRVC decode.
- **The properties:**
  - x-capstone-crvc and x-capstone-crvc-protected for CRVCB and CRVCZB: every protected U decodes in the mode, on
    the unmodified trap kernel;
  - zcb=true for ZCBB and CRVCZB, since qemu-crv defaults Zcb off;
  - LSHOFFL with none of them.
- **Platform:** the eight-stage init and stages of tests/spec8-lsrh (spec8's make.sh, all eight stages):
  - kernel 749fed2606e3ebca, fw-y, -m 963M;
  - spec/prof/capmix3.so at **shift 1**, so this is a new LSHOFFL boot, not spec8's shift-4 LSHOFF boot.
  - The driver is c4/spec8/drive3.py: drive.py with the QEMU binary, the extra CPU properties and the shift taken
    from the environment.
- **The controls,** run on this binary before this was written (c4/spec8/CTLC0..3, shift 4), each as pre-stated:
  1. LSHOFFL imagick on this binary with the properties off counts what qemu-crv counts. This checks that the port
     does not change standard decode. **PASSED:** 94,485,673, the same instruction as spec8's LSHOFF boot on
     qemu-crv, with outputs EQUAL.
  2. CRVCB gcc and imagick with the mode properties pass, with LSHOFF's outputs. **PASSED:** rc 0 and imagick's
     validation passes. Outputs EQUAL to LSHOFF's, and instructions gcc -449 (-0.002 %) and imagick -219.
  3. The same image without them fails. That shows the mode is reached and needed. **PASSED:** both stages die of
     SIGILL (rc 132, cause 2), at word 0x9508: Q0 funct3 100 [12:10] = 101, c.scc, which standard decode leaves
     illegal.
  4. LSHOFFL nab with the mode properties faults. Its image holds 606 C.FLD and 393 C.FSD, read by encoding width:
     llvm-objdump prints them as fld/fsd. **PASSED, at a C.FSD:**
     - it reads `sublet: fault cause=28 pc=0x3ff7e1144c`, rc 139;
     - the image loads at 0x3ff7e00000 against an ELF base of 0x10000, so the pc is ELF 0x2144c, `1c a1`, which is
       `c.fsd fa5, 0(a0)`;
     - in the mode that Q0 101 word is c.stc, and it faults on bounds;
     - so the first reinterpreted FP form nab executes is a store, not a load.

## Every srevroot write, and the value it writes (for the seal; step 4 itself runs the unmodified kernel)

With stage 2, bit 0 of any srevroot write is the CRVC mode and reads back. This is rtl's RTL, capstone-ariane
stage2-crvc 796af23d9 (b5b2-mwrite: `csrw srevroot, tbl|0x7ff` reads tbl|1). So every writer must write either a
page-aligned value with bit 0 clear, or exactly table_pa | crvc. Read in the final Full kernel tree and in OpenSBI
keep-tags:

| site | writes | after the kernel patch |
|---|---|---|
| switch_mm, mm/context.c:321 | next->context.capstone_root | 0 for an ordinary mm; table_pa \| crvc for a protected one |
| riscv_capstone_bind, kernel/capstone_task.c:27 | its root argument, also stored as capstone_root (:25) | the ELF loader (sublet/binfmt.h:300) passes table_pa \| crvc, with crvc from e_flags & EF_CAPSTONE_CRVC. The TRAP_TEST loader (kernel/sublet_trap.c:135, off in the final kernel) passes table_pa |
| arch_exit_mmap, kernel/capstone_task.c:50 | 0 | unchanged |
| select_root, mm/sublet-table.c:133 | c->table_pa | unchanged; its compare becomes (old & ~1UL) != c->table_pa |
| restore_root, mm/sublet-table.c:139 | the value select_root read back | an exact restore, bit 0 included |
| sublet_guard_fail, kernel/sublet_trap.c:278 | 0 | unchanged |
| the boot probe (new) | 1, then 0 | see below |
| OpenSBI keep-tags | nothing | lib/, include/, firmware/ and platform/fpga/ write no srevroot |

- **table_pa is page-aligned.** A table's root is a fresh page (SUBLET_PAGE_NEW, mm/sublet-table.h:40-48), and
  table_pa = virt_to_phys of it (mm/sublet-table.c:578). So bit 0 of capstone_root is exactly the image's flag.
- **The firmware's one 0x5c1** is T-HEAD's SHCR (thead_c9xx.h). Only platform/generic/allwinner/sun20i-d1.c uses
  it, and that file is not built for fpga/ariane.
- **The RTL's other writer,** the domain-switch path at csr_regfile.sv:2146, has no driver on Line B (rtl), and
  stage2-crvc clears crvc_q there regardless.
- **The boot probe** runs from an initcall on the boot hart, before /init: no user task, protected or not, exists yet.
  - It writes 1, which leaves srevroot_q = 0. So the protected-U predicate (cap_data_ctx[0]) is 0 throughout, and
    crvc_o = crvc_q && cap_data_ctx[0] is 0: nothing can decode in the mode between the two writes.
  - It then writes 0.
  - A bitstream without the mode reads 0 back, and the loader refuses flagged images.
- **An unflagged process never gets bit 0.** Its capstone_root is 0 for an ordinary mm, or a page-aligned table_pa
  for a protected one. Every write above is one of those, 0, table_pa | crvc, or the probe's.

## Procedure

- **Four boots, each taking and releasing the QEMU lock,** in the order LSHOFFL, CRVCB, ZCBB, CRVCZB.
  - They run after the campaign-image twin.
  - The spec8-udom boots are queued behind the same marker; whichever wrapper reaches the lock first goes first.
- **Each stage is bounded at 10,800 s,** and a stage cut in any boot is void.
- **The readers:**
  - count8.py for instructions and output hashes (the tick-line filter included);
  - perf/cclever/fetchbytes.py for fetch bytes, per stage, with the stage's command windows and its own ELF.
    fetchbytes finds each image's load delta by search, and ERRORs on no unique delta, on a shift other than 1, or
    on a PC that is not an instruction start.
- **The fetch saving** of an arm is 1 - bytes(arm) / bytes(LSHOFFL), per stage, over the benchmark's own image only.
  Libc and runtime bytes outside the image are reported separately: fetchbytes counts user PCs outside the image and
  never folds them in.

## Predictions

The prior is P1's step 4, against CRVC0 on the native route, with flags differing from B5TL1's:

| stage | P1 CRVC | P1 ZCB |
|---|---:|---:|
| perlbench | 8.47 % | 0.889 % |
| gcc (train01 on P1; t1.c here) | 14.88 % | 0.593 % |
| x264 | 2.16 % | 1.289 % |
| mcf | 6.61 % | 0.081 % |
| xz | 1.86 % | 1.259 % |
| imagick | 0.37 % | 0.982 % |
| nab | -3.53 % | 0.220 % |
| lbm | -2.98 % | 0.000 % |

1. **Instructions: each arm's count equals LSHOFFL's within 0.05 % per stage,** since compression changes only the
   encoding. Branch-range and call-relaxation choices may differ by a little. Outputs are equal in all four arms.
2. **CRVCB fetch saving within ±25 % (relative) of P1's reading** on perl, x264, mcf and xz. gcc is not predicted in
   value, since its input differs. It is predicted above 5 %, the static shrink being 10.9 %.
3. **In the mode, the FP-heavy stages lose:** nab and lbm read a negative saving, as on P1. CRVC moves their C.FLD
   and C.FSD to 4-byte forms. imagick reads between -1 % and +2 %.
4. **ZCBB fetch saving within ±0.3 points of P1's ZCB reading** on every stage but gcc, which is predicted between 0
   and 1.5 %. No stage negative beyond 0.02 %.
5. **CRVCZB:** each stage's saving is at least the larger of CRVCB's and ZCBB's, minus 0.1 points. It equals their
   sum within 0.3 points, since the two compress disjoint instruction classes. The sum is a prediction, since P1
   never measured it.

## What this decides, and what it does not

- **These readings are the inputs to the cycle band:** time saved = fetch saved × [0, 0.37] (the plan). That band,
  and the go/no-go for a stage-2 seal, are pre-registered from them before any RTL.
- **Per-image choice under option B:** a stage whose CRVC saving is negative would ship without +cap-crvc.
- I$ misses and cycles are not measured. Neither is a libc and runtime rebuilt with the features (the prebuilt ones
  are compiled without C). Silicon is not measured.
