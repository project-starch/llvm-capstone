# CVE-2026-1979 / mruby #6701 -- bytecode corruption in the pattern-matching
# JMPNOT-to-JMPIF optimization (mruby <= 3.4.0).
#
# THE COMPILER BUG (mrbgems/mruby-compiler/core/codegen.c:6632)
# The `expr in pattern` / `expr => pattern` codegen has a peephole optimization:
# when a pattern has exactly one failure exit and that jump is the last thing
# emitted, it rewrites the jump into its inverse and drops a redundant JMP:
#
#     s->iseq[fail_pos - 2] = OP_JMPIF;
#
# `fail_pos` is the position of the jump's 2-byte operand, so `fail_pos - 2`
# assumes a 4-byte OP_JMPNOT (opcode, reg, offset-hi, offset-lo) sits there.
#
# But NODE_PAT_PIN has two paths (codegen.c:4511/4524). When the pinned variable
# IS defined it emits a 4-byte OP_JMPNOT. When the pinned variable is UNDEFINED
# (`lv_idx` returns 0) it emits a 3-byte OP_JMP instead. With only 3 bytes,
# `fail_pos - 2` points one byte BEFORE the jump -- into the LAST BYTE OF THE
# PRECEDING INSTRUCTION -- and the store overwrites it with 38 (OP_JMPIF).
#
# WHY THIS TRIGGER LOOKS THE WAY IT DOES
# Three conditions all have to hold:
#
#  1. `^u` must be an UNDEFINED pinned variable -> the 3-byte OP_JMP path.
#
#  2. The match must be in STATEMENT position (its value discarded). If the value
#     is used, codegen emits OP_LOADT first, which breaks the optimization's
#     `fail_pos + 2 == s->pc` guard and no corruption happens. Hence the trailing
#     `nil` -- without it `5 in ^u` is the method's return value and nothing
#     corrupts.
#
#  3. The preceding instruction must be 2 bytes, so its single operand is the
#     last byte. `5` compiles to `LOADI_5 R1` = [opcode][dest-reg]. The store
#     therefore lands on the DESTINATION register, giving `LOADI_5 R38` --
#     an out-of-bounds register WRITE. (A wider preceding instruction, e.g. a
#     local-variable value compiling to 3-byte `MOVE`, corrupts the source
#     operand instead and only misreads.)
#
# WHY THE RECURSION
# `victim` has nregs=4, but the top-level frame is allocated with ~128 slots of
# slack, so a stray write to R38 lands inside the allocation and nothing faults.
# Recursing first marches the frame base up the VM stack until base+38 passes
# `stend`, so the write goes off the end of the allocated stack. Depth 29 is the
# first depth that crosses it; 29, 30, 31, 80 and 118 all fault.
#
# NOTE ON BUG CLASS: this reproduces as a heap-buffer-overflow WRITE (spatial),
# NOT a use-after-free. See target.md "Bug-class discrepancy".

def victim
  5 in ^u   # undefined pin -> 3-byte OP_JMP -> corrupts LOADI_5's dest to R38
  nil       # keeps the match in statement position (condition 2 above)
end

def rec(n)
  return victim if n <= 0
  rec(n - 1)
end

rec(30)
puts "done"
