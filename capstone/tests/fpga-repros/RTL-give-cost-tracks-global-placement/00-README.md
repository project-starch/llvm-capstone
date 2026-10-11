# Release cost tracks GLOBAL PLACEMENT, not the arm: a 16-byte relocation moves `give_cyc/n` by 83 %

**Status: REPRODUCED with a matched control, quantified, NOT root-caused.** Handed over as a
measurement plus one refuted mechanism, not as a diagnosed defect. Nothing here is a software bug:
the same source, built twice, measures differently because its globals landed elsewhere.

Sibling issues, so a reader who arrived with the wrong symptom is redirected now:
`../S01-image-perturbation-hang/` is the *hang* form of image sensitivity — a ~1.6 MB domain that
stops working after any perturbation, also not root-caused — and is the closest relative to this,
which is the *cost* form and does not hang. `../R35-revoked-reference-retains-authority/` is the
revocation-enforcement defect found by the same study and is unrelated to placement.
**This folder is one issue: a cost that depends on where the globals land.**

## What was observed

Three images built from identical source, running the **identical** `pressure` arm at identical
capacity from an identical invocation list. Only the link placement of the fixture's globals differs.

```
  image              m1_ret_alias          give_cyc/n      take_cyc/n
  ---------------------------------------------------------------------
  249cfda958f22f16   0x423170                71.052          72.00
  502f5ca1029904e6   0x4231b0   (+64 B)     101.049          72.00
  29d9099326304ee5   0x423180   (+16 B)     130.041          72.00
                                            \____ +83% ____/  \_ flat _/
                                              THE SIGNAL       THE CONTROL
```

**`take_cyc/n` is the control and it is on the same records.** It does not merely stay close — in the
matched pair below it is **bit-identical**, 294,912 cycles in both images. So the instrument resolves
cost to the cycle, and the movement in `give` is a property of the subject rather than drift.

## The matched pair, which is what makes this a measurement

Boot 7 ran **boot 1's invocation list on boot 2's image**: arm, capacity and position in the boot all
held, only the layout differing. Raw transcripts are in `results/`.

```
  alloc = 4096, pressure arm, same list, same capacity

  boot1  image 249cfda9   take_cyc = 294912    give_cyc = 291427    give/n =  71.149
  boot7  image 29d90993   take_cyc = 294912    give_cyc = 533056    give/n = 130.141
                          ^^^^^^^^^^^^^^^^^^                        ^^^^^^^^^^^^^^^^
                          identical to the cycle                    1.83x, +82.9 %
```

**Within an image the figure is stable; between images it is not.**

```
  WITHIN one image, across two boots :  131.788   132.213     spread  0.3 %
  BETWEEN images 16 bytes apart      :   71.052   130.041     spread 83   %
                                         \_________________/
                              a 16-byte relocation costs 250x more than
                              the entire measurement's reproducibility budget
```

## The shape of it, spatially

Nothing about the *work* changes between these runs. The same arm allocates and releases the same
number of objects of the same size. The only thing that moves is where the fixture's globals were
linked — by 16 and 64 bytes:

```
   ADDRESS SPACE                          what the run does          give_cyc/n

   0x0042_3170  [globals]  image A        ~~ identical work ~~         71.052   <-- cheap
   0x0042_3180  [globals]  image C        ~~ identical work ~~        130.041   <-- 1.83x
   0x0042_31b0  [globals]  image B        ~~ identical work ~~        101.049
                    |
                    | 16 B and 64 B apart. Same code, same arm,
                    | same capacity, same invocation list.
                    v
   0xBFF0_0000  [revocation node table]   <- the only other structure the
                                             release path walks

   take_cyc/n = 72.00 in ALL THREE  <-- the control: minting does not move at all
```

The measurement is not noisy and the arms are not different. **A pure relocation of data the program
never reads differently changes the cost of releasing it by 83 %, while the cost of allocating it does
not move by a single cycle.**

## The obvious mechanism, and why it does NOT fit

The D-cache is 32 KiB, 8-way, 16-byte lines — 2,048 lines over **256 sets**, so a 16-byte shift moves
an object by exactly one set and `set = (addr >> 4) & 0xFF`. That invites a cache-set-conflict story
between the fixture's globals and the revocation node table at `0xBFF00000` (set 0). **The ordering
refutes it as stated:**

```
  0x423170 -> set 23      give/n =  71.052     ]  set index rises  23, 24, 27
  0x423180 -> set 24      give/n = 130.041     ]  give/n goes      71, 130, 101
  0x4231b0 -> set 27      give/n = 101.049     ]  NOT MONOTONE
```

A single aliasing pair between one global and one table line would give a spike at one set and a flat
floor elsewhere; three points cannot distinguish that from several other shapes, and they do not lie
on any monotone function of the set index. **So the dependence on the image is established and the
mechanism is not.** Do not write "cache-set conflict" into anything citing this folder.

## What would settle it

**The RTL lane priced this and 256 images is over-buying** (answers from source at `054cea69b`):

- The node table base is a **compile-time `localparam`** — `CAP_REVNODE_MEM_BASE = 56'hBFF0_0000`
  (`core/include/ariane_pkg.sv:591`), consumed at `ex_stage.sv:1165-1166` as
  `base + {36'd0, node_addr[15:0], 4'd0}`. So node *i* sits at `0xBFF00000 + i*16`, one node per line,
  and with the base's low 20 bits zero, **set = i mod 256 with the table origin at set 0**. The set
  arithmetic above is sound and there is no fencepost.
- **What `give` touches that `take` does not is a serialized, data-dependent miss chain.** `REVOKE_NODE`
  reads node `revoke_index` and only then learns the next address (`capstone_rev_node.anvil:15`, with
  the read serialized behind `mem_wait_flag` at `:92-94`), so hop *k+1*'s address is unknown until hop
  *k* returns: **zero memory-level parallelism**, r dependent 16-byte reads in r distinct sets. `take`
  is the opposite — a fixed handful of addresses known up front. That is why `take` coming back
  bit-identical is **the matched control proving the sensitivity is in the walk**, not a null result.
- **The response has a period, and the period is computable.** The walk's set footprint is fixed by the
  **mint-id sequence** — a property of the fixture, not of its link address. Sequential minting gives a
  contiguous run of sets and a *step* response; round-robin over k slots gives stride k and **period k**.
  Either way the sweep needed is ~the period, not 256.

**So: derive the period in simulation first, then spend ~period images on the board** with `take_cyc` as
the matched control. The Verilator build instantiates the real `wt_dcache` from the same config package,
so the conflict effect itself reproduces there. Caveat kept explicit: `S12_MEM_DELAY` is a flat
period-16 sawtooth rather than DDR, so **simulation owns the shape and the period; the board still owns
the magnitude.**

The sweep, once its size is known:

```
  build N images that walk m1_ret_alias across all 256 sets in 16-byte steps
  (a pad global, or DOMAIN_WINDOW), run ONE arm at one capacity, plot give_cyc/n
  against set index:

     flat                  -> placement is not the variable after all; look elsewhere
     periodic with p = 256 -> set conflict, and the period names the structure it hits
     one isolated step     -> a single aliasing pair, identifiable by its address
     something else        -> the shape is the finding
```

`take_cyc/n` rides along as the per-image control for free: it must stay at 72.00 throughout, and any
image where it moves is excluded from the plot rather than explained.

## The cache-set-conflict mechanism is REFUTED, twice, by arithmetic (2026-09-22)

The section above says the set ordering does not fit. Two further checks, both free, close it off —
and the second one refutes the mechanism rather than merely failing to support it.

**Refutation 1 — every placement is INSIDE the hot window, so nothing flips in or out.** The rotation
re-uses a fixed, small set of indices: `give(i)` pushes slot *i*'s node onto the LIFO free head and the
very next `take(i)` pops the head, which is the node just pushed. The **generation** increments while
the **index does not**, and the address is `base + index*16` with the generation masked off, so the
node footprint is **16 fixed cache lines, permanently**. The harness reports the build's allocating-op
count directly — `fixture_nodes = 2 * M1_LIVE - 1`, confirmed across the sweep (3/7/15/31/127 for
M1_LIVE 2/4/8/16/64). At M1_LIVE = 16 that is 31, and with `head` starting at 3 and bumping:

```
   ids consumed by the build   :  3 .. 33      (31 allocating ops)
   the 16 rotating slots hold  : 18 .. 33      -> hot sets 18 .. 33

   globals 0x423170 -> set 23   give/n =  71.052    INSIDE
   globals 0x423180 -> set 24   give/n = 130.041    INSIDE
   globals 0x4231b0 -> set 27   give/n = 101.049    INSIDE
```

All three are interior. A 16-byte shift never moves a global out of the window, so "in or out of the
hot set range" cannot produce the spread.

**Refutation 2 — the mechanism predicts ZERO where the data swings 83 %.** The cache is 8-way, so
evicting a hot node line needs real pressure in its set, not merely a line landing there. The globals
block in that image is **76,832 B = 4,802 lines** over 256 sets:

```
   4,802 lines / 256 sets = 18.8 global lines per set
   ways per set           =  8
   -> EVERY set is over-subscribed 2.3x by the globals alone
```

So the set a line leaves and the set it enters were **both already over-subscribed before it moved**.
Under the serial-dependent-chain reading — cost = hops x (hit or miss), placement deciding which hops
hit — the node lines are evicted in *every* placement, every hop misses always, and the predicted
response to a 16-byte shift is **zero**. The matched pair measures **83 %**. A mechanism that predicts
flat where the data swings is refuted.

**What the mechanism must instead depend on.** Not how many lines map to a set, but **which** lines are
hot and **in what order** they are touched. One RTL fact constrains that search: `REVOKE_NODE` advances
strictly via `walk_next := node_in.next`, so the traversal is in **chain order, not id order**. With
LIFO reuse those diverge — the 16 ids are reissued in push order while the chain threads them in link
order — so the touched-address sequence is a *permutation* of the 16 that depends on the give/take
history rather than a scan. If order is the variable, that permutation is where to look.

## What to do about it TODAY, before anyone settles the mechanism

The mechanism is open, but the reporting rule that follows from it is not, and it is enforceable now:

```
   about to quote a cost figure measured on this platform?
                 |
       +---------+---------+
       |                   |
   take_cyc/n          give_cyc/n
   (minting)           (release)
       |                   |
   72.00 in all        71 / 101 / 130 across three images
   three images        that differ by 16 bytes
       |                   |
       v                   v
   SAFE to quote       QUOTE ONLY WITH ITS IMAGE HASH
   unqualified         - name the image in the same sentence
                       - never compare two images' absolute figures
                       - growth WITHIN one image is still a result
                       - the spread between images (83 %) is 250x the
                         measurement's own reproducibility (0.3 %)
```

The distinction is not cosmetic: `take` is bit-identical at 294,912 cycles across the pair, so this is
a property of the **release** path specifically, not of the harness or of the board's variability.

## Why this matters beyond the study that found it

**No release-cost figure measured on this platform may be quoted without naming its image.** The M1
study records this as a correction to its own numbers, and R1's release-cost bundle inherits the
constraint. `take_cyc/n` (minting) is unaffected — 72.00 in all three images — and is safe to quote.

## Provenance

Found by the M1 node-reclamation study, 2026-09-19 ladder, on bitstream
`caplifive_m1_054cea69b.bit`. Full bundle:
`experiments/results/M1/2026-09-19-maxret-ladder/` in the nested-allocators-paper tree; the relevant
excerpt is `results/m1-summary-excerpt.md` here.

**The bitstream's timing does not close (WNS −8.307, 90,379 failing endpoints).** Nothing in this
folder separates a design property from an artefact of that build, and no measurement here was taken
on a timing-clean image.
