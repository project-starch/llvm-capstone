# C-63 at run time: C++ exceptions in a trap-route domain (plan, 2026-11-25, nothing built)

C-63's compiler half is fixed (ISSUES C-63, 2026-11-20). Under a capability personality the exception pointer arrives
in `C10` and the selector in `X11`. A landingpad selects, and `.gcc_except_table` and `.eh_frame` are emitted. What is
missing is everything at run time. No unwinder (`_Unwind_RaiseException`, `_Unwind_Resume`) and no C++ ABI runtime
(`__cxa_throw`, `__cxa_begin_catch`, `__gxx_personality_v0`) exists for a domain, so C++ with exceptions compiles and
does not link. The runtime lane placed it on the trap SDK (origin/trap/eval), the route the B5b-2 bitstream runs.
None of the paper's workloads is C++ with exceptions.

## What the code generator gives the unwinder (read 2026-11-25, clang 29027126f39b, `-funwind-tables`)

A function that saves callee-saved capabilities emits this prologue:

    cincoffsetimm sp, sp, -128     .cfi_def_cfa_offset 128
    sd   ra, 112(sp)               .cfi_offset ra, -16
    stc  s0, 96(sp)                .cfi_offset s0, -32
    stc  s1, 80(sp)                .cfi_offset s1, -48
    ...
    cincoffsetimm s0, sp, 128      .cfi_def_cfa s0, 0

- **The CFI uses the integer register numbers.** `s0` to `s11` are saved as 16-byte capabilities by STC, and DWARF
  says only "at CFA-32". The unwinder therefore has to know, as an ABI rule, that a callee-saved GPR's save slot
  holds a capability. It must restore the slot with LDC, never LD. An LD would restore the cursor and lose the tag,
  and the next use of `s0` as a base would raise 24.
- **`ra` is an integer** (SD, 8 bytes). Calls are JAL/JALR. CJALR is illegal under Sublet, and the trap route checks
  no fetch. So the return address and the PC need no capability.
- **`sp` is a capability** (the stack reference). The CFA is computed as an offset from it, so the unwinder must
  derive the CFA with CINCOFFSET from the frame's `sp` or `s0` capability, never rebuild it from an integer.
- **Not yet checked:** FP registers (`fs0`-`fs11`, plain FSD/FLD, so integer as on RISC-V); `.eh_frame`'s pointer
  encodings (pcrel sdata4, read as integers relative to the image); and whether any callee-saved slot holds an
  integer under some codegen path (an `ra`-like GPR saved by SD), which would break the "slot is a capability" rule.

## The pieces

1. **libunwind** (in-tree `libunwind/`): `Registers_riscv` holds `uint64_t` GPRs and its save/restore assembly
   (`UnwindRegistersSave.S`, `UnwindRegistersRestore.S`) uses SD/LD.
   - A capability variant must hold 16-byte GPRs.
   - Its save must use STC.
   - Its restore must use LDC for the GPRs and keep `ra` integer.
   - `getReg`/`setReg` (and the `unw_word_t` interface) must carry capabilities, while the DWARF expression
     machinery stays in integers. How `uintptr_t` lowers on this target was not checked here.
   - `DwarfInstructions::stepWithDwarf` must apply the "slot is a capability" rule above.
   - Its `.eh_frame` lookup (`dl_iterate_phdr` or `__eh_frame_hdr` registration) needs a capability covering the
     image's `.eh_frame` and `.eh_frame_hdr`. The trap loader grants one block covering text and data with the
     cursor at the base (binfmt.h, `dom_grant`), which is enough if the runtime derives from it.
2. **libc++abi** (in-tree `libcxxabi/`): `__cxa_*`, the personality and the LSDA reader.
   - It needs libc++'s headers (`<exception>`, `<new>`, `<typeinfo>`), so a header-only libc++ configuration for
     the capstone target.
   - `__cxa_allocate_exception` goes through mallocng, which is fine.
   - The personality reads the LSDA through a pointer that must be a capability into the image.
   - Thread-local exception globals use the SDK's TLS.
3. **The SDK:**
   - capstone-cc compiles C only today (`compile_flags` and its checks are C), so it needs a C++ path that keeps
     the profile checks.
   - Link order: libc++abi and libunwind after the runtime, before libc.
   - `-fno-exceptions` stays the default until 1 and 2 pass.

## Acceptance, in order

- **Link:** a `try`/`catch` C++ program links against the trap SDK with no undefined symbols.
- **Positive control (QEMU, the trap kit):** a throw caught one frame up and five frames up, with callee-saved
  capabilities live across the throw. After the catch, every `s` register still dereferences (tag intact). This is
  the check that would catch an LD-restored slot.
- **Negative control:** the same program built with a restore that uses LD for the GPRs must fault with cause 24 at
  the first post-catch dereference. This shows the control can fire.
- **Rethrow and cleanup:** `throw;`, a destructor run during unwinding, `std::terminate` on an uncaught exception
  (exit by SIGABRT, not a capability fault).
- **No regression:** C images are byte-identical with the C++ runtime present in the SDK. The runtime links only
  when C++ objects are present.

## Open questions for the semantics owner

- Is "every callee-saved GPR slot is a capability" an ABI rule the compiler guarantees? Today it is a reading of one
  prologue, not a contract. A DWARF vendor register range for capability registers, or an augmentation letter, would
  make it explicit.
- Where does the unwinder get its capability for `.eh_frame`: from gp (image-gp), or from a symbol the runtime
  publishes?
