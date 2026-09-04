# S-12: relocating the store cured it (4/4) but did NOT discriminate — the ADDRESSES do some of that work

## The experiment and its honest verdict

The folder's proposed discriminator: relocate `Index *pIdx = 0;` past `pWC = &pWInfo->sWC;` so the
null-capability store leaves the faulting window. Built with the trap handler on, so a fault
returns; sha `8176deb0ca99e277`.

Result: **4 / 4 draws completed** (`SLT-SUMMARY … completed=1`), control green, no trap. The wedge
disappears.

**That is the uninformative direction, exactly as pre-registered.** Both live accounts predict a
cure when the adjacent capability store is removed, and a source edit moves every address, so the
result is additionally confounded with layout. It confirms the adjacent store is required — which
the one-byte ladder already showed by changing its source register — and separates nothing.

What the compiler actually produced is worth recording, because it is not quite what the
discriminator assumed:

    baseline                            relocated
    104868  movc a4, zero               104854  stc  a4, -0x5a0(s0)   <- an STC on a4 survives, 5 back
    10486c  stc  a4, 0x0(a5)     <--    104860  sw   a4, 0x0(a5)      <- now a PLAIN store, no rd alias
    104870  ldc  a4, 0x0(a0)            104864  ldc  a4, 0x0(a0)
    104874  cincoffsetimm a4, a4        104868  cincoffsetimm a4, a4  <- fault site

So the adjacent **capability** store is gone while an STC sourcing `a4` remains five instructions
back. Under the ordering-escape account that surviving STC could still have been live at the load's
issue; it evidently was not.

## The observation that DOES narrow it, and it was free

From the baseline disassembly:

* the store writes **`s0-0x120`**
* the load reads **`s0-0x70`**
* **176 bytes apart — 11 capability granules.**

The write-buffer-forward account rests on `wbuffer_hit_oh` being **word-granular against a 16-byte
capability**, i.e. a partial mismatch *within* one granule. That does not explain a hit across 176
bytes. No other store in the window aliases the load's slot either (`-0x5a0`, `0x0(a3)`).

**So a data forward from this store to this load requires the match logic to ignore high address
bits, which is a far stronger defect than the word-granularity artifact the account was built on.**
That disfavours the write-buffer account for THIS pair without needing another board draw.

Stated as a caveat, not a refutation: I am not the auditor here, the hit logic should be read at
source before this is leaned on, and the account could still hold if the forward involves a store
outside the window entirely.

## Where the two accounts now stand

| account | status |
|---|---|
| **commit-stall ordering escape** — `we_gpr` asserted while `commit_ack` withheld, releasing younger instructions before their producer produced | **LEADING.** Address-independent, so the 176-byte gap costs it nothing. Consistent with every board result including this one. |
| **wrong-address write-buffer forward** | **DISFAVOURED** by the address distance, though not excluded. |

Neither is proven. The one-byte ladder, the fence/NOP pair and this relocation are all consistent
with both; what separated them was meant to be this experiment, and it did not.

## What would actually discriminate, for whoever picks this up

Keep the store adjacent AND keep its source register `a4`, but change **only its address** so it
aliases the load's slot (or moves further from it). Address is irrelevant to the ordering escape and
decisive for a data forward. That is a one-word patch to the store's offset, it does not move any
other instruction, and it is the clean version of the experiment this one tried to be.
