
---

## RETRACTION — "give costs ~71 retained and ~103 dropped" does not survive boot 2

After boot 1 this bundle's author reported a clean regularity: `give_cyc/n` ≈ 71 whenever the old alias
is retained and ≈ 103 when it is dropped, across five arms and two boots, with the release arm's own
phase boundary as the controlled case. **Boot 2 breaks it.** Its release phase 1 retains every alias and
gives at **131.788**, against boot 1's pressure arm at **71.052** — same branch of the harness, same
first-invocation position, ~43,000 allocations each, `snap_every` 4,096 both.

**Two variables moved together and the generalisation was made anyway.** The arm differs (pressure vs
release) *and* the image differs. The images differ by a single 16-byte start-line field, and every
global is shifted by exactly **16 bytes — one 16-byte cache line, and one set in a 32 KiB, 8-way,
16-byte-line D-cache**:

| symbol | `249cfda958f22f16` | `29d9099326304ee5` |
|---|---|---|
| `m1_ret_alias` | `0x423170` | `0x423180` |
| `m1_ring_alias` | `0x4cc370` | `0x4cc380` |
| `m1_tmp` | `0x4cc470` | `0x4cc480` |

**The control has since run, and it confirms the layout.** Boot 7 ran boot 1's list on boot 2's image —
arm, capacity and invocation position all held fixed, only the layout differing. The same `pressure`
arm read **130.041** where boot 1 read **71.052**.

Three images now carry the **identical** `pressure` arm, and the give cost tracks the image:

| image | `m1_ret_alias` | `pressure` give/n | `release` phase-1 give/n |
|---|---|---|---|
| `249cfda958f22f16` | `0x423170` | **71.052** | 71.827 |
| `502f5ca1029904e6` | `0x4231b0` | **101.049** | — |
| `29d9099326304ee5` | `0x423180` | **130.041** | 131.788 *and* 132.213 (two boots) |

**Within an image the cost is the same whatever the arm**, and it reproduces across boots (131.788 vs
132.213). **Between images it moves by 83 %.** So `give_cyc/n` is a property of **global placement**, not
of retaining or dropping, and these numbers must never be quoted as properties of the design without
naming the image. The plausible mechanism is cache-set conflict between the node table and the
fixture's globals in a 32 KiB, 8-way, 16-byte-line D-cache, where a 16-byte shift moves an object by one
set; that mechanism is **not** established here, only the dependence on the image.

**`take_cyc/n` is the robust quantity** — 72.00 in all three images, and ~72 in every retaining regime
across every boot, rising to ~87 only in a cleared phase 2.
