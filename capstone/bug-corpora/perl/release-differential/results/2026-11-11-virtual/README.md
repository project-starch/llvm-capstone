# Perl 5.36.3 on virtual Capstone with the one-operand lifetime instructions, 2026-11-11

The two virtual arms of [`../2026-10-11-virtual/`](../2026-10-11-virtual/README.md), run again after the lifetime
instructions changed: `CREVOKE` takes the object alone. The SV-head patch
(`ports/perl/sublet/patches/5.36.3/0001`) was converted: each head is a `CDERIVE` child of its arena, without the
arena's intermediate lifetime in the arena's first slot, and `plant_SV` ends the recorded child with
`capstone_cap_release`. CheriBSD was not run again.

## The arms

| arm | what | image |
|---|---|---|
| `virtual-malloc` | Perl as released on the virtual profile's musl mallocng | 708d23a2ca31 |
| `virtual-nested-pools` | the same with `PERLD_SUBLET=1` | 8d744f0291c4 |

Both images carry port patches 0001-0012 and are built by `ports/perl/musl/build-perl-domain.sh` with a corpus SDK
(`CAPSTONE_SDK_CORPUS=1`, effective `-fno-builtin`), which `runners/virtual/run-virtual.py` requires since the
2026-10-11 record. Both controls passed on both arms. One VM per arm, one process per case, 600 s per case.

## Rows

| case | layer | virtual-malloc | virtual-nested-pools | equal to 2026-10-11 |
|---|---|---|---|---|
| 01 | nested | fault 24, `Perl_pp_iter` | fault 25, `Perl_pp_iter` | yes, both |
| 02 | nested | exit 255, Perl's own "attempt to copy freed scalar" | fault 25, `Perl_sv_setsv_flags` | yes, both |
| 03 | system | completed, [PASS] | the same | yes, both |
| 04 | system | fault 25, `memmove` (site) | the same | yes, both |
| 05 | nested | completed, [FAIL] | fault 25, `Perl_cv_undef_flags` | yes, both |
| 06 | -- | completed, [FAIL] 5 of 29 | **fault 25, `Perl_regfree_internal`**, after the same [FAIL] 5 of 29 | stock yes; protected **no** (was completed) |
| 07 | system | fault 25, `S_regcppop` (site) | the same | yes, both |
| 08 | spatial | fault 28, `Perl_pp_gvsv` | the same | yes, both |
| 09 | nested | fault 24, `Perl_newATTRSUB_x` | fault 25, `Perl_newATTRSUB_x` | yes, both |
| 10 | system | fault 25, `memmove` (site) | the same | yes, both |
| 11 | -- | completed, [FAIL] | the same | yes, both |

Read by the registered rules of 2026-10-11, the counts do not move: system allocator 04, 07 and 10 caught by both
arms; nested allocator 02 and 05 caught by the protected arm only. 21 of the 22 cells have the outcome, the cause and
the function of the 2026-10-11 record.

## The one cell that differs: case 06 on the protected arm

The 2026-10-11 record took 06 out of both counts: it is not a temporal defect. `S_free_codeblocks` walks entries whose
`src_regex` was never written, so what it frees is whatever the uninitialized memory holds. On 2026-10-11 the protected
image completed it. Now it prints the same five wrong results and then faults with cause 25 in `Perl_regfree_internal`,
which is not one of the case's `fault_sites`. The fault was not traced to the entry it came through.

This is not the instruction change. Perl built from unmodified dev (f8d6cb8bc) with the two-operand patch and run on
the QEMU dev pins reads the same fault, cause 25 in `Perl_regfree_internal` (`control-dev/`, cases 05 and 06, both
controls passed). The reading therefore moved between the 2026-10-11 platform and current dev, with the contents of
uninitialized memory. The cell stays outside both counts, for the reason 2026-10-11 gave.

## Platform and inputs

- Tree: branch `one-operand-crevoke-crenew` at 59201bdf2, QEMU at ea57497908. The SV-head patch is 822a151fc302....
- Platform: QEMU 2d5a71078904..., launcher 0f8e0f903725..., module e47bca62addf... loaded without parameters (pinned,
  one context), `capstone_vm up --profile virtual --exact-bounds --cpu-prop x-capstone-test-csrs=true`. Each bundle's
  `inputs.json` has the hashes; it does not record the CPU property.
- `control-dev/`: image f64da049ebf7, QEMU 35529815bf4b..., module 1f2ac8580711... (the landing gates' copy with 32
  contexts, `x-capstone-supervisor-slots=32`).
- The lifetime instructions in the images: stock 6 `CDERIVE` and 3 `CREVOKE` words, the C library's mallocng;
  protected 46 and 6. No `CRENEW`.

## Caveats

- One run per case (N = 1). The two arms and six other sessions of the same campaign ran on the host at the same
  time; no case reached its 600 s bound.
- The 2026-10-11 caveats about the stock port's own faults in Perl's core tests and about `$x = *foo; *x = $x` were
  not measured again here.
