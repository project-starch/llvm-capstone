# C4 continued: loop-carried copies, the sink pass, and spill round trips

Follows `20-11-2026_18-39-20_c4-move-census-per-benchmark.md`. The strict round trips counted there were
negligible. This note counts the shape that census missed, tries one fix for it, and records why that fix is
restricted and what it measured.

## Loop-carried copies

A loop-carried copy is a move inside a loop body whose destination was read earlier in the same iteration (the
value is carried round the loop) and whose source was written earlier in it (the next value was made in another
register). The census uses linear loop bodies only, so it is a lower bound. Its positive control is xz's `memcpy`
byte loop, which has two copies per byte and reads exactly 2 x 84,036,988. Profiles and arms are as in the C4 note
(per-instruction capmix, every window; perlbench's riscv-base side cleaned of its `natperl` driver, which runs at
overlapping addresses there).

Capstone minus riscv-base, as a share of riscv-base's instructions:

| benchmark | loop-carried copies | without `memmove`/`memcpy` (replaced by dev's runtime) |
|---|---:|---:|
| perlbench test-sw | +0.647 % | +0.391 % |
| xz test | +0.595 % | about +0.26 % |
| mcf test | +1.122 % | |
| lbm test | +0.432 % | |
| gcc train01 | +0.165 % | |
| nab test | -0.011 % | |

Three shapes:
- **(a)** A pointer incremented before its last use in the iteration: the byte loops, and `S_regmatch`'s. Plain
  RISC-V clang emits the same `mv`; it is generic LLVM and shows against riscv-base only because riscv-base is
  GCC-built.
- **(b)** A counter split across two registers by the allocator: mcf's `price_out_impl`, `mv a3, s10` at the
  depth-2 header and `addi s10, a3, 1` at its latch, 154 M copies (0.58 % of mcf's instructions). Deferred:
  on mcf the time goes to memory and walks.
- **(c)** lbm: the exit test reads the old index, so the increment cannot move below it. That is the IV-shape item.

## The sink pass, and why it is restricted

`-capstone-enable-sink-loop-increment` (lane compiler/sink-loop-increment, default off) moves a loop increment below
the last use of the value it replaces, so the old and new values do not overlap and the copy coalesces away. The
instructions are the same, and nothing is accessed through the new value.

Its first version sank a trapping increment (CIncOffset, 26 on a sealed or uninitialised base) when the next
effectful instruction was a load through the old base, on the argument that the load would raise 26 in its place.
That argument was RETRACTED the same day. On silicon a scalar access's base type is checked only in M-mode with
capability mode on (`load_store_unit.sv:1390-1393` at 6d0e754d5; R-30 and R-31 record it as measured inert in
domains). In a domain the load through a sealed or uninitialised base goes ahead, so sinking would turn
trap-before-read into read-then-trap (Q-36). Decision: a trapping increment moves only when its base is proven
usable, meaning derived in the function from a stack slot or a global's base through offsets, narrowings, copies
and PHIs, never through a load or an argument. ADDI cannot trap and moves freely.

Increments moved per benchmark (static, measured flags):

| benchmark | first version (loads settle the order) | restricted (proven-usable trapping ones) |
|---|---:|---:|
| perlbench | 249 | 40 (27) |
| gcc | 475 | 140 (48) |
| imagick | 75 | 19 (15) |
| nab | 60 | 18 (18) |
| xz | 10 | 6 |
| mcf | 1 | 1 |
| lbm | 0 | 0 |

Neither version touches the hot problem functions: `S_regmatch`'s static loop-carried copies are 657 off and 657 on,
`cse_insn`'s 153 and 153.

Twins of the first version (pre-registered at 808e43deb78b, kit-unq, QEMU 5b5563063547e5ac):
| workload | off | on | on vs off | floor | band |
|---|---:|---:|---:|---:|---|
| perl test | 86,163,257,062 and 86,163,892,503 (image 316d502b5caf5406) | 86,130,597,568 | -0.038 % | 0.0007 % | -0.40..+0.01 %, inside |
| gcc train01 | 12,246,627,859 | 12,245,504,655 | -0.009 % | 0.003 % (earlier pair) | -0.20..+0.01 %, inside |
| xz test (12 commands) | 49,640,016,062 and 49,640,018,670 | 49,639,965,360 | -0.0001 % | 0.000005 % | -0.05..+0.01 %, inside |

specdiff PASS on every output (perl 2/2 with 560 hashes, gcc 1/1, xz 12/12). lbm's image is identical on and off.

Reading. The measured version is not the one that can land: the restricted pass moves about a sixth of perl's
sites and a third of gcc's. Even the measured version gains at most 0.04 % of an instruction count, and nothing on
the hot problem functions. The pass stays on its lane, not landed; its value is the lit-pinned shape (a fill loop
of five instructions an iteration becomes three) and the census above.

## Spill round trips

A spill round trip is a store to a stack slot followed in the same iteration by a load from that slot. Its share
of each image's instructions:

| benchmark | Capstone | riscv-base |
|---|---:|---:|
| lbm test | 4.736 % (`fsd`/`fld` in `LBM_performStreamCollideTRT`, about 5.5 an iteration) | 4.261 % |
| perlbench test-sw | 1.617 % (`S_regmatch` 688 M) | 2.226 % |
| mcf test | 1.025 % (`spec_qsort`) | 1.798 % |

GCC's riscv-base spills as much or more. lbm's "seven spills an iteration" are mostly there on riscv-base too, and
Capstone's excess is about +0.5 % of lbm's instructions. Not a lever.

## Instrument slips caught on the way, before any figure was reported

- The quadratic copy census was replaced by a linear one. The first linear version counted a destination read
  after it had been rewritten, and read +103.8 M on xz where the quadratic one read +294.2 M. The corrected version
  reproduces the quadratic reading exactly.
- Function attribution first took objdump's `.Lpcrel_hi` labels for functions. They split loops and moved xz's
  reading; function symbols now come from the symbol table, with local labels excluded.
