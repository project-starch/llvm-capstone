# C-22 — `c ? (__int128)-1 : k` drops the condition and returns `k`

**SILENTLY WRONG CODE, not a crash.** That makes this the most dangerous of the four
compiler issues filed here; C-19, C-20 and C-21 all announce themselves.

## Reproducer

    define i128 @mixed_sign_arms(i32 %c) {
      %t = icmp ne i32 %c, 0
      %r = select i1 %t, i128 -1, i128 7
      ret i128 %r
    }

emits

    mixed_sign_arms:
        li a0, 7
        cjalr zero, 0(ra)

The condition is gone. The function returns 7 for every input, where it must return
-1 when `c != 0`.

## Where it goes wrong

Not in the select lowering -- that code never runs. DAGCombine rewrites
`select c, -1, 7` into the standard form `or(sext(c), 7)`, which is correct. Type
legalisation then produces

    t21: i128 = sign_extend_inreg (any_extend t18), ValueType:i1
    t14: i128 = or t21, Constant:i128<7>

and between there and instruction selection the `or` disappears, leaving ISel with

    CopyToReg ... Constant:i128<7>

So the fold is in the i128 `or` / `sign_extend_inreg` handling, and it discards the
whole first operand.

## Not caused by the C-21 fix

Checked, not assumed: the same wrong output appears with the C-21 change stashed and
llc rebuilt. C-21 is about materialising a constant ARM of a select; this input never
reaches that path.

## How it was found

Writing a regression test for C-21 and putting a case in it whose expected output
looked wrong. The test was reduced to shapes that exercise C-21 alone, and this was
split out rather than left in as a failing check -- a test that fails for two reasons
tells you about neither.
