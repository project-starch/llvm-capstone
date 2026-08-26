# The three codegen gaps that are not `long double`

Of the 32 archive members the LTO build drops for failing codegen, 27 are the
`long double` math family and go away with `long double` at 64 bits. These three
do not, so they are the ones that still need compiler work:

    src_stdlib_qsort.o     Cannot select: i128 = xor t6, Constant:i128<-1>
    src_network_dn_comp.o  Cannot select: i128 = xor t85, Constant:i128<-1>
    src_regex_regcomp.o    Cannot materialize arbitrary >64-bit constants

## They are two problems, not three

**`xor` of a zero-extended i64.** Reduced to six lines, and the reduction is the
useful part -- `xor i128, -1` on its own selects fine, so the gap is in the
*extended* shape:

    define i128 @zext_then_not(i64 %x) addrspace(200) {
      %z = zext i64 %x to i128
      %n = xor i128 %z, -1
      ret i128 %n
    }

    LLVM ERROR: Cannot select: t5: i128 = xor t3, Constant:i128<-1>
      t3: i128 = zero_extend t2

`any_extend` fails the same way when the result is consumed rather than returned.
Both qsort and dn_comp are this.

**The oversized constant is a REFUSAL, not a gap.** `ret i128 <something over 64
bits>` produces "Capstone PureCap: Cannot materialize arbitrary >64-bit constants
as capabilities; capabilities are unforgeable", which the target says on purpose.
So the fix for regcomp cannot be in the materialisation -- it has to be upstream,
in whatever produces a 128-bit constant from 64-bit source code. The IR contains
no such literal; it appears during codegen.

## Where the i128 comes from

All three fail at llc `-O0`, `-O1` and `-O2` alike, so the codegen level is not the
variable: the i128 is already present in the bitcode clang emitted at `-O1`. This
is the re-widening that `build-musl-capstone.sh` already records -- the optimiser
widens 64-bit integer arithmetic to the POINTER width, which on this target is
128 bits and therefore the capability carrier.

That is why the old `-O0` rescue worked: recompiling the SOURCE at `-O0` never
creates the i128. It is also why the rescue cannot be carried into an LTO build --
a native `-O0` object brings its own descriptor fragment, which is the multi-TU
slot collision.

## What would close them

* the `zext`/`anyext` + `xor` pattern: an ISel pattern, and the six-line case above
  is enough to iterate on;
* regcomp: find the fold that turns 64-bit code into a 128-bit constant. Not
  reduced further here -- that needs llvm-reduce over `@regcomp`, which is many
  llc runs and was not worth spending while a QEMU suite held the machine.
