# The five allocator-corpus defects inside the running memcached (2026-10-02)

**Question.** `bug-corpora/memcached/allocator-repros` holds five historical memcached allocator
defects, case 03 being CVE-2018-1000127. Until now their only evidence was from the freestanding
component harness, so nothing said what the *server* does with them. Does each defect still occur
inside the real memcached, and does Sublet inside memcached's own allocators stop it?

**Pre-registration.** Fixtures 12–16 and their predictions (`app/host/safety-expect.txt`, the block
headed "FIXTURES 12-16") were pushed to lane branch `corpus-in-app` in 40f5f333095b before the first
boot. Nothing in the predictions has changed since.

## Verdict

**35 of 35 runs as pre-registered** — 5 fixtures × (1 control boot + 3 boots per slabsublet mode).

| fixture | corpus case | `sublet` (control) | `slabsublet0` | `slabsublet1` |
|---|---|---|---|---|
| 12 | 00 read buffer copied after it went back to its cache | returns `c00101` | returns `c00101` | **temporal fault** |
| 13 | 01 IO list walked while the body frees and reissues the entry | returns `d00101` | returns `d00101` | **temporal fault** |
| 14 | 02 tail repair assigns `refcount = 1` over a holder's reference | returns `e0015c` | returns `e0015c` | **temporal fault** |
| 15 | 03 **CVE-2018-1000127**, the `unsigned short` refcount wraps | returns `f0012e` | returns `f0012e` | **temporal fault** |
| 16 | 04 an unlocked decrement loses a concurrent get | returns `1000191` | returns `1000191` | **temporal fault** |

Reading the three columns: on the control the defect happens and nothing notices; with per-unit
bounds but no revocation it still happens; with revocation on free every one of the five is a trap.

**The returns are evidence that the defect occurred, not merely that the program survived.** Each
mark encodes `(reuse << 8) | value`:
- 14–16 read back the *new* item's byte — `0x5c`, `0x2e`, `0x91`, each case's own `OTHER` value —
  where the holder had written the fixtures' own `0xa7`, at the same address (`same-address=1`), so
  the chunk really was freed and reissued under a live holder. Without the premature free
  `slabs_alloc` would have handed back a different chunk, making `same-address=0` and the byte
  `0xa7`;
- 12 reports `damaged=1`: the command bytes copied out of the returned read buffer differ, because
  the cache's free-list link was written over them;
- 13 reports `walked=1/3`: the walk ended after one entry, because the reissued object was zeroed by
  its new owner and the stale link read as end-of-list.

**The CVE fixture's own output**, from the `slabsublet1` boot that then faulted:

```
MCAPP-FIX 15 a cursor=cc0fff50 bounds=[cc0fff00,cc0fffa0) len-from-cursor=80
MCAPP-FIX 15 took=65536 refcount-now=2 wrapped=1
MCAPP-FIX 15 freed-while-held=1
MCAPP-FIX 15 same-address=1
MCAPP-FIX 15 target=cc0fff50
```

and QEMU's diagnostic `cincoffset with an UNTAGGED rs1 -- pc=0xc0247db8 ... val=0xcc0fff50`, whose
value is the published target and whose pc is the fixture's own probe.

**The hooks fired, counted by the adapter itself** (one extra mode-0 boot with
`MC_SLAB_SUBLET_REPORT=1`, same image; the 35 runs above did not set it, so they carry no hook
count of their own):

| fixture | `pages` | `chunk_releases` / `chunk_reuses` | `object_releases` / `object_reuses` |
|---|---|---|---|
| 12 | 0 | 0 / 0 | 4 / 1 |
| 13 | 0 | 0 / 0 | 2 / 1 |
| 14, 15, 16 | 1 | **1 / 1** | 1 / 0 |

The one chunk release and one reuse in 14–16 are the premature free and the reissue; their zero in
12 and 13, against object activity there, is the layer split. Some object releases are the server's
own — every connection takes and returns objects from `cache.c` — so these counts bound the
fixtures' contribution from above, not exactly.

## Where each fixture probes, against the corpus's own choice

- **13 probes the object's offset 0, where `01_*/case.c` probes `&io->io_next`**, the link one level
  in. The link read still happens — it is what produces `walked=1/3` — but on `slabsublet1` the
  probe traps first, so "the link read traps" is inference from the whole object's alias being
  revoked, not something observed here.
- **14–16 probe the payload where the corpus probes the item header.** That is deliberate and makes
  the evidence stronger, not weaker: the header's first bytes are part of a pointer and
  unpredictable, while the payload byte is a known fill, so the returned value is a pre-registrable
  constant.

## What this does and does not say about the CVE

**Does:** memcached 1.6.45 still stores `refcount` as an `unsigned short` and still increments it
with a bare `++`, so driving it past 65536 still wraps it past the holders and the next release
still frees a held item — and with Sublet in the slab allocator that freed chunk's alias is dead,
so the holder traps instead of reading another item's data.

**Does not:** this is not an exploit of a shipped 1.6.45 server. The CVE's fix was a ceiling in the
multiget consumer (`memcached.c`), and fixture 15 does not go through that consumer — it drives the
counter directly. Reaching the defect the way the CVE did needs a version before 1.4.37. That is the
next stage of this work, and until it runs, nothing here should be written as "we caught
CVE-2018-1000127 in a vulnerable memcached".

## The reductions, stated

Each fixture reproduces its case's **allocator-level** shape and performs the premature free itself.
It does not drive memcached's consumer path. Specifically, and matching what the corpus's own
PROVENANCE files leave out:

- 14's real branch needs `-o tail_repair_time`, the clock, the LRU tail and an exhausted class; the
  two statements are performed directly.
- 15 does not go through the multiget loop, and does not show issue #271's self-linked hash chain or
  the remaining 65534 releases.
- 13 and 16 are races upstream; the interleaving is written out in program order, as the corpus
  writes it.
- 12 and 13 use their own cache instance rather than a worker's live `rbuf_cache`, so the fixture
  cannot disturb the connection serving it.

Four of the five defects (00, 01, 03, 04) are fixed in 1.6.45; only 02 is live at the pin. These
fixtures therefore demonstrate the **defect class and the allocator's response**, which is what the
arms differ on, not reachability in the shipped server.

## The component corpus, re-run the same day as a cross-check

The fixtures say what the server does; the corpus's own runners say the cases themselves still behave
as recorded. Both arms this host can run were re-run on 2026-10-02 and are recorded in each case's
`case.json` status (that corpus gitignores `results/` by policy — the case file is the record):

- **native**, all five cases: fixed arm `VERDICT FIXED` first, then buggy arm
  `VERDICT DEFECT-REPRODUCED` with `damage=1`. No guest, no lock.
- **capstone-domain**, all five cases: **10/10 arms passed** — spatial completes, sublet faults at
  `mc_defect_read` with cause 24 at the expected pc.
- **negative control: 10/10 oracles fired**, none reporting a pass on an input that never ran its
  case. Without this the 10/10 above would not be evidence.
- **CheriBSD and PoisonCap: not re-run.** That platform's SDK, purecap sysroot and image are not on
  this host. Their 2026-09-21 record stands as the latest for them, and the case files say so rather
  than leaving the gap silent.

## Controls these runs do NOT contain

Three gaps, named so nothing here is read as more than it is:

- **No in-app fixed arm.** Nothing in these 35 runs shows the non-defective ordering returning
  `walked=3`, `damaged=0` or byte `0xa7`. The matched fixed arm is the component corpus's native run
  of the same day (all five `VERDICT FIXED` with `damage=0`, then `VERDICT DEFECT-REPRODUCED` with
  `damage=1`) — a different harness, cited as such.
- **No arm in which `slabsublet1` returns**, so these runs alone cannot show the trap is specific
  rather than indiscriminate. That control is the 2026-10-01 mode-1 oracle run, whose transcript was
  identical to native with `chunk_releases=20` — a **different image, on a different day**
  (`../2026-10-01-qemu-slab-sublet/`).
- **The control arm is N=1** (one boot, against three per slabsublet mode) and differs from the
  slabsublet images in three ways, not one: `MC_CAPSTONE_SLAB_SUBLET` undefined, a different SDK
  (`HEAP_LOG 27` only for slabsublet), and no `-m 48 -o no_slab_reassign`. Its objects are narrowed
  to exactly the unit, which is the useful direction: it says narrow bounds alone do not help, and
  the missing ingredient is a revoke at the allocator-internal transition. The one-variable
  comparison is `slabsublet0` vs `slabsublet1` — same image hash, one environment variable.

## Limits

- **`toolchain-fresh` could not run** in either build (`cannot check: no build.ninja`, warning
  rc=2). A gate that did not run is not a gate that passed. It does not endanger this result — the
  tested images are pinned by hash in `SHA256SUMS` and verified with `sha256sum -c` — but it is
  recorded rather than left implicit.
- The server flags are recorded as the runner passed them, duplicate `-m` included:
  `-l 127.0.0.1 -p 21299 -U 0 -m 64 -t 4 -m 48 -o no_slab_reassign` (the last `-m` wins). Identical
  on both slabsublet arms.
- QEMU only. The slabsublet arms need `CAPSTONE_REV_NODES=16777216`; silicon's pool is 65,536.
- `-m 48 -o no_slab_reassign` on the slabsublet arms, as for every run of that arm.
- The temporal faults are QEMU untagging a revoked capability on reload (ISSUES Q-11); on silicon
  that behaviour is recorded as open.
- The control uses the plain `sublet` image (a3a3d5b6…), which does not carry patch 0006; the two
  slabsublet arms are the same image (9f6467ff…) differing only in `MC_SLAB_SUBLET_MODE`.

Files: `result-lines.txt` (every run), `SHA256SUMS` (both images, verifiable from `$MC_WORK`).
