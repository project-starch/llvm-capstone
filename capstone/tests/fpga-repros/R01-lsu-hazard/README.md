# A load addressed through a register misses pending stores to other addresses

We've been chasing a divergence where four of our seven benchmark kernels either hang or return
wrong values on the board while running correctly under QEMU. It is now isolated to a single
behaviour, with a five-line reproducer, and we could not work around it in software. Sending it
over in case it's useful — and in case we've misread something.

## The behaviour

> A load whose address arrives **through a register** — either a register-carried capability or a
> register-computed offset — does not observe **pending stores to other addresses**.

Concretely, this loop should run 5 times and returns 1:

```c
rh_a[0] = 5; rh_a[2] = 0;
long j = opaque(1);                 /* j in a register, opaque to the compiler */
while (rh_a[j-1] > 0 && n < 50) {
    rh_a[2]   = rh_a[2] + 1;        /* a store to a DIFFERENT location */
    rh_a[j-1] = rh_a[j-1] - 1;
    n++;
}
/* board: n == 1     QEMU and native: n == 5 */
```

It reads as though the condition's load never sees the body's update once another store is in
flight, so the loop exits after one pass.

## Both ingredients are required

Each of these was run on the board, one domain per clean boot, every image QEMU-verified first:

| variant | result |
|---|---|
| register index `[j-1]`, **no** other store | **5** correct |
| literal index `[0]`, **with** the other store | **5** correct |
| **register index + other store** | **1** wrong |
| same, store placed *after* the decrement | **1** wrong |
| plain `[j]` instead of `[j-1]` | **1** wrong |
| re-run of the failing case on a separate boot | **1** wrong (deterministic) |

So it isn't the index arithmetic, and it isn't store ordering. Neither ingredient alone does it.

## What the emitted code actually does

Reading the disassembly sharpens this a lot. The array capability is loaded once and a second one
is derived from it:

```
ldc        a1, 0(gp)        ; the array capability
cincoffset a4, a1, a4       ; a4 = a1 + j*4 -- a SECOND capability into the same object
lw         a5, -4(a4)       ; condition load  rh_a[j-1]  via a4
lw/sw      a7, 8(a1)        ; the other store rh_a[2]    via a1
lw         t0, -4(a4)       ; reload          rh_a[j-1]  via a4   <-- returns stale
```

The variant that **works** (literal index) uses `0(a1)` and `8(a1)` — one capability register with
constant offsets. Every variant that **fails** uses two capability registers derived from the same
object, load through one and the intervening store through the other.

So the sharper statement is: *an intervening store through one capability register appears to make
a later load through a different capability register miss an earlier store to its own address* —
though the two addresses are distinct and both capabilities are in-bounds derivations of the same
object. That would also explain why the pointer-walk variants fail (a pointer walk creates exactly
such a second capability register) and why cache-line separation changes nothing.

As far as we can tell the code is ordinary PureCap — deriving a second capability with
`cincoffset` and using both is normal, both are in-bounds, and the addresses do not overlap. But
this is exactly the point where we would rather be told we are wrong.

## It doesn't need a loop

Hoisting the value out — `long v = rh_a[j-1];` before the loop — made the loop run **zero** times:
that single register-indexed load returned 0 where 5 had just been stored. The loop only makes the
effect easy to see.

## Seven workarounds, none of which helped

| attempt | result |
|---|---|
| `fence rw,rw` immediately before the load | still wrong |
| `fence rw,rw` after every store in the body | still wrong |
| hoist the value into a register | worse (0 trips) |
| make the other store register-indexed too | still wrong |
| place the two locations 64 B apart | still wrong |
| walk a pointer so the load is `lw 0(p)` | still wrong |
| do both accesses through pointers | still wrong |

The fences not helping is what makes us think this is address disambiguation in the load path
rather than a memory-ordering question — but that's our inference, not something we can see.

A read-only pointer walk with no stores in the loop is correct, so loads on their own are fine.

The one shape that *is* reliably correct is a load at a compile-time-constant offset from a
capability freshly re-derived from the cap table (`ldc gp[i]` then `lw imm(cap)`). We can't build
a general workaround on that, because a dynamic array index has no compile-time-constant base —
which is why we ran out of software options.

## Why it matters to us

It accounts for exactly which of our kernels work. The ones that pass touch a single location per
iteration; the ones that fail all do something of the form `C[i*N+j] += A[…] * B[…]` — a
register-indexed load with a store to another location outstanding. That covers a matrix multiply,
CoreMark's matrix kernel, a CRC table loop and an insertion sort.

## Running it

```bash
tar xzf capstone-lsu-hazard-repro.tar.gz && cd capstone-lsu-hazard-repro
# One image per clean boot (a second domain at the same entry VA within one boot
# hangs regardless, unrelated to this).
#   transfer images/ladder_perf_ctl and images/rawhazard5.dom to the target
#   ./ladder_perf_ctl rawhazard5.dom
```

It prints a `DEBUG` line of raw slot values. Correct hardware gives `5 5 5 5 5`; we see
`5 5 1 1 1` — the first two are the single-ingredient controls, the last three are the failing
combinations.

`rawhazard6.dom` is the seven-workaround sweep and `rawhazard7.dom` the pointer-walk variants, if
the detail is useful. `src/` has everything needed to rebuild rather than trust our binaries. The
domains are built `-O1`, `-capstone-gp-captable`, shrink off, `-fno-jump-tables`, `+m`.

## Two questions

1. Is this a known characteristic of this LSU, and is there a code pattern we should be emitting
   to avoid it?
2. Is there anything wrong with the way we're deriving these addresses in the first place? We'd
   rather find out that this is ours than have you chase it.

## Separately, and much smaller

A `delin` executed in domain code on a capability loaded from the gp cap table wedges the board —
recoverable only by power cycle. We think that one is probably ours (we were emitting a redundant
`delin`; the entry glue already delins each cap-table entry before storing it, and removing ours
fixes it with no downside). The only thing we'd flag is that the failure mode is a full wedge
rather than a catchable trap. Happy to send that reproducer too if it's of interest.
