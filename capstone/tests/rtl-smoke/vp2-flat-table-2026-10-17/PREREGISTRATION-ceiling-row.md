# Pre-registration — the ceiling row (node-contract at 70,000 on the flat arm)

**Written and pushed BEFORE the run.** Cite this commit from whatever reports the result. Nothing
here may be edited afterwards; corrections go in the result.

## What is being read

`node-contract` in its default `growth` mode on the FLAT arm only, `CAPSTONE_NODE_COUNT` at its
default 70,000. Not paired with the paged arm: the paged table has no cap, so it is not a control
for a ceiling reading. This row records the behaviour silicon will show for a load that wants more
live structural nodes than the flat table holds — a cpython-like load, whose measured high-water is
41,601 and therefore inside the ceiling, which is why the question is worth answering now rather
than after a surprise.

## The arithmetic, computed and not estimated

| quantity | value |
|---|---:|
| flat capacity, fixed | 65,536 |
| header ids (0 and 1) | 2 |
| usable ids | 65,534 |
| `CAP_REV_TABLE_RESERVE` | 256 |
| next-unused when `available <= 256` | 65,280 |
| **successful handles before pressure** | **65,278** |
| requested live nodes | 70,000 |
| shortfall against the reserve point | 4,722 |
| excess over usable ids even at zero reserve | 4,466 |

## Predicted reading if the flat mode is behaving

1. Round 0's loop issues handles until next-unused reaches **65,280**, i.e. after **65,278**
   successful `sublet_handle` calls, at which point `available = capacity - next + free_count`
   falls to 256 and **node pressure (kind 5)** fires.
2. A **software collection** runs and reclaims **nothing**. `node-contract` keeps every structural
   node live by construction: its own comment at line 54 says "Losing the REV handle does not retire
   its structural node", so `sublet_clear` does not make a node collectable.
3. The allocation therefore cannot be satisfied by any collection strategy — 70,000 exceeds the
   65,534 usable ids by 4,466 even if the reserve were zero — so the request fails through the
   ordinary resource path.
4. **`NODE_LIVE:0:70000` is never printed** and **`NODE_GROWTH_OK` is never printed.** The domain
   reads resource exhaustion: **cause 30 (INSUF_RESOURCES)**.

So the recorded silicon behaviour is: a load needing more than 65,278 concurrently live structural
nodes gets pressure, then a collection that cannot help, then cause 30 — and the number 65,278 is
the usable headroom for one context on vp2.

## What a different reading would mean

* **The test COMPLETES** (`NODE_GROWTH_OK`): either the cap did not bind — a kit mismatch, the
  thing this whole exercise exists to prevent — or structural nodes *are* being retired by
  `sublet_clear`, contradicting the source's own comment. Both are findings, and neither is a pass.
* **Pressure at a materially different count**: the available formula is not what the module and
  QEMU agree on, and the pressure rule would not read identically on both kits.
* **A cause other than 30** (24, 25, 29): the exhaustion is not travelling the resource path, which
  is what the flat mode's no-growth branch is supposed to guarantee.
* **A hang or no reading**: treated as an instrument failure, not as a ceiling reading.

## The separate collection row, for contrast

The collection check is `malloc-contract` with `CAPSTONE_MALLOC_COUNT` raised until the PAGED arm
collects at least once, run at the same count on both arms, comparing `collections`,
`nodes_reclaimed`, `nodes_retired` and the test's own pass. That row has the paged arm as its
reference and does not involve the ceiling. Bundle 5 cannot serve as it: measured peak high-water
31, collections 0 on both arms, so it never creates the triggering condition.
