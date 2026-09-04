# C-20 — `__builtin_ctz` on a 32-bit value crashes the Capstone backend

**A COMPILER bug, not a silicon defect.** Filed here rather than in `tests/fpga-repros/`,
which is for suspected RTL defects handed to the hardware side. Sibling issue a reader may
have arrived looking for: C-19, the cap-init block split, lives in
`../C19-capinit-block-split-oob/`.

## Reproducer

    unsigned g(unsigned x) { return __builtin_ctz(x); }

    clang -target capstone64-unknown-elf -ffreestanding -nostdlibinc -std=c99 -O0

    LegalizeDAG.cpp:1352: SelectionDAGLegalize::LegalizeOp(SDNode *):
      Assertion `(Res.getValueType() == Node->getValueType(0)
                  || Node->getValueType(0) == MVT::Glue)
                 && "Type mismatch for custom legalized operation"' failed.

`./run.sh` reproduces it and prints a verdict; `./run.sh workaround` shows the one
mitigation found. Neither needs a board or QEMU.

The failing node, from `-debug-only=isel`:

    t11: i32 = cttz_zero_undef t10

## What is and is not established

**Established.** `__builtin_ctz` and `__builtin_ctzl` crash. `__builtin_clz` and
`__builtin_popcount` compile. `-mattr=+zbb` makes it compile.

**NOT established: the root cause.** A first hypothesis was that the guard at
`CapstoneISelLowering.cpp:509-511` is at fault, because it marks `ISD::CTTZ` and
`ISD::CTTZ_ZERO_UNDEF` for `MVT::i32` as `Custom` whenever `hasCTZLike() && is64Bit()`,
while the CTLZ block twenty lines below additionally requires `hasStdExtZbb()` --

    if (Subtarget.hasCTZLike()) {
      if (Subtarget.is64Bit())                                    // <- no Zbb condition
        setOperationAction({ISD::CTTZ, ISD::CTTZ_ZERO_UNDEF}, MVT::i32, Custom);

    if (Subtarget.hasCLZLike()) {
      if (Subtarget.is64Bit() && Subtarget.hasStdExtZbb())        // <- has one
        setOperationAction({ISD::CTLZ, ISD::CTLZ_ZERO_UNDEF}, MVT::i32, Custom);

That asymmetry is real and matches which builtin crashes. **It is also not the fix.**
Adding the `hasStdExtZbb()` condition to the CTTZ block and rebuilding leaves the crash
exactly as it was, on the same node. So whatever is custom-legalising this to the wrong
type is not switched off by that action alone, and the hypothesis is recorded as REFUTED
rather than left standing because it looked convincing.

Anyone picking this up should start from the failing node above, not from the guard.

## Why it matters

Found in a ten-minute feasibility probe of JerryScript, before any porting work. Of
JerryScript's 200 core files, 134 failed to compile for this target: 132 for two missing
shim headers (`inttypes.h`, `ctype.h`), one for a missing `nextafter`, and one -- this --
for a compiler crash. `jerry-core/ecma/base/ecma-helpers-number.c` uses `__builtin_ctz`
in `ecma_integer_multiply` to turn a multiply by a power of two into a shift.

It is not JerryScript-specific. Any C that counts trailing zeros hits it, which includes
most bit-manipulation and many allocator free-list scans.

## The workaround, and why it is not adopted here

`-mattr=+zbb` compiles the file, and the emitted instruction is `ctzw`. But **no build
script in this tree passes `+zbb`**, and whether the silicon implements Zbb was not
established while writing this. Adding it would change the ISA every domain is compiled
for on the strength of a crash workaround. `run.sh workaround` demonstrates it and says
the same thing.
