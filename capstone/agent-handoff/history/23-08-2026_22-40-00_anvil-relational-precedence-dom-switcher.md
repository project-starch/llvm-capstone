# Anvil relational operators bind LOOSER than `||`/`&&` — a second collapsed guard, in the domain switcher

Found 2026-08-23 while checking two incidental observations handed back from the SQLite lane.
One of them (SHRINK) turned out to be fine, and proving it fine is what produced the corrected
precedence model — which then located a second live instance of the S-11 collapse.

## The refined model

S-11's README currently says the model is *"Anvil binds logical and bitwise operators tighter
than the comparison operators."* **That is too broad, and `SHRINK` refutes it.**

`capstone_flu_unit.anvil:182` is an unparenthesised `==`/`!=` chain next to `||`:

```
if(rd_in.metadata.cap_type==NOT_CAP ||rs1.metadata.cap_type!=NOT_CAP)||(rs2...!=NOT_CAP){
```

and generates **correctly** — `capstone_flu_unit.anvil.sv:2328-2338` is three genuine 3-bit
comparisons, each yielding one bit, OR'd together:

```systemverilog
:2328  $1461 = $1459 == 3'd0     // comparison FIRST, then the OR
:2332  $1465 = $1463 != 3'd0
:2333  $1466 = $1461 || $1465
:2337  $1470 = $1468 != 3'd0
:2338  $1471 = $1466 || $1470
```

So equality binds **tighter** than `||`, exactly as in C. Every observation in the tree fits a
narrower rule instead — tightest to loosest:

| | operators | evidence |
|---|---|---|
| tightest | `&` | `perm&3'd6!=3'd6` generates correctly (S-11 README) |
| | `==` `!=` | SHRINK, above |
| | `\|\|` `&&` | — |
| **loosest** | **`<` `>` `<=` `>=`** | S-11's SEAL, and the dom switcher below |

**Relational operators have the LOWEST precedence in Anvil.** That is the whole defect class.
Anything of the form `a < b || c` parses as `a < (b || c)`, and since `b` is normally a nonzero
constant, `(b || c)` folds to `1` and the guard becomes `a < 1`, i.e. `a == 0`.

## The second instance: `capstone_dom_switcher.anvil:115`

```
if *cur_idx < 7'd3 || (*cur_idx > 7'd8 && *cur_idx < 7'd57) {
    call process(64'd16, 1'b1, 7'd66)      // 16 bytes, metadata_en = 1  (capability)
} else {
    call process(64'd8,  1'b0, 7'd66)      // 8 bytes,  metadata_en = 0  (scalar)
}
```

Generated, `core/capstone_dom_switcher.anvil.sv:261-270` — the collapse happens **twice**:

```systemverilog
:263  $13  = cur_idx_q
:264  $14  = 7'd8 && $13        // (8!=0) && (cur_idx!=0)  ==>  cur_idx != 0   [1 bit]
:266  $16  = $14 < 7'd57        // {0,1} < 57              ==>  CONSTANT 1
:267  $17  = $11 > $16          // cur_idx > 1
:268  $18  = 7'd3 || $17        // 3 is truthy             ==>  CONSTANT 1
:269  $19  = $9  < $18          // cur_idx < 1             ==>  cur_idx == 0
```

`$19` drives the branch directly (`:775` taken, `:736` not-taken). So a guard meaning
*"index 0-2 or 9-56"* means **`cur_idx == 0`**.

**Matched pair, same file, differing by exactly one thing.** The sibling four lines down is the
same operator on the same operands *without* a trailing `||`, and it is correct:

```systemverilog
:411  $160 = cur_idx_q < 7'd3       // from `if *cur_idx < 7'd3`  -- CORRECT
:269  $19  = cur_idx_q < $18        // from `... < 7'd3 || (...)` -- COLLAPSED
```

## Impact: real, and currently LATENT — the branch is dead

The intended policy is documented in-tree at `core/store_unit.sv:385`:
*"(capstone_dom_switcher.anvil: `process(64'd8, 1'b0, ...)` for idx 3..8 and 57..66)"* — so
indices 0-2 and 9-56 are meant to move as 16-byte capabilities. Under the collapse only index 0
does; **every other register would move as 8 bytes with `metadata_en = 0`, dropping tag and
metadata.**

That would be severe, and it is not happening, because the guard sits under
`if *commit_req.is_full` and **`is_full` is structurally zero on this RTL**:

- `capstone_unit.anvilh:367` defaults `dom_switch_is_full = 1'b0`;
- `:383` sets it only from the `is_full` parameter of `create_result_pack_domain_switch`;
- both callers pass `1'b0` — `capstone_dyn_unit.anvil:267` (CALL) and `:302` (RETURN);
- no SystemVerilog writes it; `ex_stage.sv:1191` only forwards it.

An exhaustive `grep` for `is_full` across `core/` returns no other writer. So the buggy arm is
unreachable today. **The bug is latent, and it is armed for whoever first implements a full
domain switch** — at which point capability registers silently become scalars, with no
exception and no diagnostic.

Checked for staleness: the generated `.sv` is 11 s newer than its `.anvil` source, same build,
so this is the current generation and not a leftover.

## Status of the two observations that started this

- **SHRINK — NOT A DEFECT.** Generates correctly, and is the positive evidence for the model above.
- **SPLIT — DOWNGRADED, and my first reading of it was wrong.** I said it "does not raise when
  both operands are NOT_CAP". It does raise: `capstone_dyn_unit.anvil:117`'s `&&` is false in
  that case, but `:120`'s `(rs1 != LINEAR) && (rs1 != NONLIN)` is then true, so it raises
  `UNEXPECTED_CAP_TYPE` (**27**) instead of `UNEXPECTED_OPERAND` (25). A wrong exception code,
  not a missing raise. **NO SPEC BASIS CHECKED** — I have not read what the spec requires here.

## What should happen next

The fix for both S-11 and this one is parentheses, and the *class* is worth a sweep: any
unparenthesised relational sharing a line with `||`/`&&`. That search over `anvil_build/`
returns exactly two hits that are not already fully parenthesised — `capstone_flu_unit.anvil:167`
(S-11) and `capstone_dom_switcher.anvil:115` (this one). Everything else in the tree already
wraps its relationals, which is why the class has stayed almost entirely invisible.
