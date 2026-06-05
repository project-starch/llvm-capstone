# Capstone backend compiler fixes

Known backend bugs surfaced during CoreMark PureCap bring-up.  Each is currently
worked around in `capstone/benchmarks/coremark/build-coremark-capstone.sh` with a comment
pointing to the root cause.  These need proper fixes in the LLVM backend.

## Bug catalog

| Bug | Current workaround | Root fix needed |
|-----|--------------------|----------------|
| `cincoffset rd≠rs1` with LINEAR rs1 consumes the cap; compiler hoists the offset computation above a loop, so the first iteration consumes the LINEAR cap and subsequent iterations crash | `CAPSTONE_DELIN(ptr)` applied manually before each affected pointer; `-O0` for files where the compiler would hoist | Backend: after instruction selection, identify `cincoffset` uses where rs1 is a LINEAR cap and rd≠rs1; insert an automatic `delin` on rs1 before the first such use |
| Tail-call lowering emits `cjalr ra, imm(rs1)` (a CALL that sets `ra = pc+4`) instead of restoring `ra` and jumping | `-fno-optimize-sibling-calls` on affected files | `CapstoneISelLowering`: detect sibling-call scenario; emit `ldc ra, N(sp)` + `cincoffsetimm sp, sp, frame` + `cjalr zero, 0(ra)` instead of a CALL |
| Jump tables use scalar `lw` base load in cap_mem mode; `lw` requires `rs1.tag=1` but the GP-derived table pointer is scalar | `-fno-jump-tables` on affected files | `CapstoneISelLowering` / `CapstoneAsmPrinter`: lower jump tables through `ldc` (capability load) or materialize table entries as PC-relative capabilities so the table base is tagged |
| `va_list` arg-pointer stored via `sd` (scalar), not `stc` (capability); reloaded via `ld` → tag=0 → any memory dereference crashes in cap_mem mode | Assembly trampoline `ee_printf_asm.S` bypasses `va_list` entirely by forwarding `a0-a7` directly to `ee_printf_impl()` | Capstone ABI / LLVM clang front-end: `va_start` must store the variadic argument pointer as a capability (`stc`), not a scalar integer |
| Function prologue emits `cincoffsetimm rd≠rs1` on `sp` (rd=s0, rs1=sp), which consumes `sp` (LINEAR); subsequent `ldc`/`stc` using `sp` as base crash on `sp.tag=0` | Hand-written assembly entry point `coremark_domain_entry.S` replaces the compiler-generated `domain_main` | Backend frame lowering: prologue must use `cincoffsetimm rd==rs1` (in-place cursor update, no consumption) for the stack-pointer frame allocation |

## Related files

- `capstone/benchmarks/coremark/build-coremark-capstone.sh` — every workaround has a comment block
- `capstone/benchmarks/coremark/coremark_domain_entry.S` — hand-written entry point (prologue bug workaround)
- `capstone/benchmarks/coremark/ee_printf_asm.S` — va_list bug workaround
- `llvm/lib/Target/Capstone/CapstoneISelLowering.cpp` — tail-call and jump-table lowering
- `llvm/lib/Target/Capstone/CapstoneFrameLowering.cpp` — prologue/epilogue emission
