# perlbench and gcc: the instruction gap to riscv-base by class, and the C3 gate

The runtime lane asked for the lever to be sized before C3 (register allocation): perlbench's and gcc's image
instruction gap over riscv-base decomposed by instruction class, each class's share of the gap with its top three
functions. C3 goes ahead only if coalescable capability moves reach 0.5 % of instructions on perlbench or gcc.

Result: they do not, once the estimate is calibrated against GCC's own output. The largest classes are global-address
formation (`scc gp`) on perlbench and the add family on gcc, with `scc gp` second.

## Instrument

- Profiles: shift-1 (per-instruction) capmix, every command window, mapped by one load offset per image instance.
  Capstone: arm FRCDKMSa3, the paper's Full arm. riscv-base: the native RVAL profiles, GCC-built; perlbench's has
  its `natperl` driver subtracted PC by PC (C4 note).
- Classes: `perf/cclever/levers.py`'s classifier (14 Nov census). Paired so that a capability operation sits beside
  its integer counterpart: loads (`ldc` with `ld`/`lw`/...), stores, the add family (`cincoffset*` with `add`/`addi`),
  moves (`movc` with `mv`).
- Executed PCs that decode to no instruction are an error; none occurred. Each table sums exactly to its gap.

## The gap by class (classes of at least 0.5 % of the gap)

**500.perlbench_r test-sw (FRCDKMSa3)**: capstone 84,330,511,919, riscv-base 74,992,812,111, gap +9,337,699,808 (+12.45 %)

| class | Capstone | riscv-base | delta | share of gap | of riscv-base | top functions by delta |
|---|---:|---:|---:|---:|---:|---|
| scc gp (global address) | 2,277,692,920 | 0 | +2,277,692,920 | 24.4% | 3.04% | S_regmatch +305M, Perl_pp_nextstate +152M, Perl_yyparse +142M |
| moves (movc, mv) | 5,328,427,064 | 3,071,190,887 | +2,257,236,177 | 24.2% | 3.01% | S_regmatch +525M, Perl_leave_scope +222M, memmove +220M |
| add family (cincoffset*, add, addi) | 13,033,735,194 | 11,222,477,562 | +1,811,257,632 | 19.4% | 2.42% | __libc_free +327M, __libc_malloc_impl +196M, memmove +184M |
| bounds (shrink / shrinkto) | 931,057,866 | 0 | +931,057,866 | 10.0% | 1.24% | Perl_yyparse +161M, Perl_leave_scope +111M, __libc_malloc_impl +85M |
| loads (all) | 16,506,069,016 | 15,745,906,317 | +760,162,699 | 8.1% | 1.01% | __libc_free +216M, Perl_yyparse +131M, memmove +88M |
| conditional branch | 12,547,747,720 | 12,036,292,974 | +511,454,746 | 5.5% | 0.68% | memmove +204M, __libc_free +107M, memset +89M |
| constants (li / lui / auipc) | 7,420,633,140 | 6,916,582,327 | +504,050,813 | 5.4% | 0.67% | __libc_free +144M, __libc_malloc_impl +117M, S_regmatch +54M |
| other integer | 11,745,489,706 | 11,260,123,747 | +485,365,959 | 5.2% | 0.65% | S_regcppop +598M, __libc_free +212M, S_regcppush +167M |
| sp adjust (addi sp / cincoffsetimm sp) | 1,183,356,672 | 1,112,267,998 | +71,088,674 | 0.8% | 0.09% | __unlock +50M, Perl_save_I32 +9M, S_expect_number +2M |
| return / indirect jump | 1,027,510,857 | 959,065,110 | +68,445,747 | 0.7% | 0.09% | __libc_malloc_impl +20M, memmove +19M, malloc +18M |
| other capability op | 66,427,394 | 0 | +66,427,394 | 0.7% | 0.09% | __libc_free +36M, __libc_malloc_impl +20M, __malloc_allzerop +6M |
| jump | 1,401,805,139 | 1,865,357,139 | -463,552,000 | -5.0% | -0.62% | Perl_leave_scope +10M, Perl_sv_clear +6M, Perl_sv_setsv_flags +5M |

**502.gcc_r train01 (FRCDKMSa3)**: capstone 12,221,817,514, riscv-base 10,848,633,963, gap +1,373,183,551 (+12.66 %)

| class | Capstone | riscv-base | delta | share of gap | of riscv-base | top functions by delta |
|---|---:|---:|---:|---:|---:|---|
| add family (cincoffset*, add, addi) | 1,380,823,220 | 1,064,086,210 | +316,737,010 | 23.1% | 2.92% | __libc_free +18M, ggc_alloc_stat +11M, bitmap_elt_ior +11M |
| scc gp (global address) | 221,479,525 | 0 | +221,479,525 | 16.1% | 2.04% | record_reg_classes +8M, df_ref_record +7M, cselib_invalidate_regn +6M |
| moves (movc, mv) | 901,299,048 | 687,524,109 | +213,774,939 | 15.6% | 1.97% | bitmap_elt_ior +12M, bitmap_set_bit +11M, memset +11M |
| loads (all) | 2,915,768,157 | 2,767,967,755 | +147,800,402 | 10.8% | 1.36% | bitmap_element_allocat +26M, bitmap_elt_ior +23M, __libc_free +12M |
| constants (li / lui / auipc) | 1,128,278,292 | 993,776,067 | +134,502,225 | 9.8% | 1.24% | bitmap_elt_ior +9M, __libc_free +8M, cselib_invalidate_regn +6M |
| bounds (shrink / shrinkto) | 128,687,053 | 0 | +128,687,053 | 9.4% | 1.19% | record_reg_classes +7M, ggc_alloc_stat +6M, __libc_malloc_impl +4M |
| conditional branch | 1,613,463,477 | 1,541,669,660 | +71,793,817 | 5.2% | 0.66% | memset +15M, bitmap_element_allocat +13M, bitmap_elt_ior +7M |
| stores (all) | 1,628,564,200 | 1,567,312,718 | +61,251,482 | 4.5% | 0.56% | bitmap_elt_ior +24M, bitmap_element_allocat +22M, bitmap_ior_into +12M |
| other integer | 1,378,127,293 | 1,342,935,698 | +35,191,595 | 2.6% | 0.32% | __libc_free +12M, aligned_alloc +7M, __heap_mapping +6M |
| jump | 220,727,084 | 204,877,192 | +15,849,892 | 1.2% | 0.15% | bitmap_elt_ior +4M, bitmap_element_allocat +4M, bitmap_ior_into +2M |
| sp adjust (addi sp / cincoffsetimm sp) | 290,056,425 | 281,138,162 | +8,918,263 | 0.6% | 0.08% | bitmap_element_allocat +8M, __unlock +3M, bitmap_elt_ior +1M |
| return / indirect jump | 223,696,198 | 215,116,905 | +8,579,293 | 0.6% | 0.08% | bitmap_element_allocat +4M, __libc_malloc_impl +1M, malloc +1M |

Two classes have no riscv-base counterpart and are capability cost outright: `scc gp`, one capability derivation
per global access, and bounds (`shrink`/`shrinkto`). The add family and loads are net of what capability addressing
saves: perlbench's integer adds fall by 2.95 G while `cincoffset*` adds 4.76 G.

## The C3 gate: coalescable capability moves

A `movc` counts as coalescable when it is not ABI-constrained (not argument setup, return value or return setup) and
its source is overwritten before it is read again, scanning forward within its basic block. The allocator could then
have given source and destination one register.
- perlbench test-sw: 0.54 % of instructions. `memmove` and `memcpy` hold 0.12 % of that, and dev's runtime replaces
  their byte loops.
- gcc train01: 0.87 %, led by `bitmap_elt_ior`, `note_stores` and `walk_tree_1`.

Calibration: the same test on plain `mv`. Dead-source moves are a property of any allocator's output, not only of
Capstone's.

| | Capstone `movc` | Capstone `mv` | Capstone, all moves | riscv-base `mv` (GCC) |
|---|---:|---:|---:|---:|
| perlbench | 0.54 % | 0.41 % | 0.95 % | 0.93 % |
| gcc | 0.87 % | 0.49 % | 1.36 % | 1.15 % |

The excess over riscv-base is about +0.02 % on perlbench and +0.21 % on gcc, under the 0.5 % gate on both. C3 does
not go ahead. The concrete cases recorded earlier stay as test cases if allocation work is taken up later:
perlbench `S_regmatch`'s uncoalesced capability shuffle (about 80 M moves) and mcf's split counter (154 M).

## Next lever

The largest class on perlbench is `scc gp`, at 24.4 % of the gap and 3.04 % of instructions: `S_regmatch` +305 M,
`Perl_pp_nextstate` +152 M, `Perl_yyparse` +142 M. On gcc it is second (16.1 % of the gap, 2.04 % of instructions),
after the add family (23.1 %).

What is already done for global bases: the `%pcrel_lo` fold (04ed898afc58), one shared base per constant pool
(a464d645f34c), and the dropped DELIN (C-27). Nothing yet reuses or hoists the per-access `scc gp` within a function.
The programme names that as "bounded-global hoisting".

Tools: `c4/gap.py` (`--paired`), `c4/coal.py` and its mnemonic-parameterised copy, all in the compiler lane's work tree.
