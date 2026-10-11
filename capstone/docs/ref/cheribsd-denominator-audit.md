# The CheriBSD denominator: which corpus cases actually carry that arm (2026-10-03)

The nested-allocators paper prints a comparative headline — CheriBSD's deployed heap revocation stops
**0** of the reproduced corpus — and backs it with a sentence that is stronger than a zero:

> "Every comparator now runs the whole denominator, so a zero in `\targetCheriDetected` or
> `\targetSpatialDetected` is a measured miss and never an absent integration."
> — `paper-nested-allocators/appendices/b-target-results.tex:136`

This document is the per-corpus check of that sentence, so the claim can be defended or narrowed from
evidence rather than from memory. **It reports numbers; it does not edit the paper.**

## Answer

~~**22 of the 57 counted cases carry a committed CheriBSD measurement.**~~ **Superseded 2026-10-07.**

**The three programs the paper's target evaluation uses now carry 53 measured rows of 54**, each on
one vehicle (image `0cb16209…`), each with `cheribsd-abi` and `cheribsd-bounds` passing in the same
boot, and each temporal row with a **revocation control that faults**:

| program | cases with the arm | measured | caught | bundles |
|---|---:|---:|---:|---|
| **memcached** | 11 | **11** | 1 | `allocator-repros/results/20261006-cheribsd`, `…-case8`, `plain-heap-repros/results/20261007-cheribsd` |
| **tshark** | 24 | **24** | 2 | `wmem-repros/results/20261007-cheribsd-revocation-control`, `plain-heap-repros/results/20261007-cheribsd` |
| **FFmpeg** | 19 | **18** | 4 | three `results/20261006-cheribsd` bundles |

The one unmeasured row is `ffmpeg/plane-repros/00`: it links a real native `libavutil.a`
(`runners/run-native.sh:16`), so a purecap reading needs a purecap `libavutil` this project has never
built. The blocker was **tried**, not assumed.

**CPython (20), PostgreSQL (8) and SQLite (19) still carry none of any kind**, and none of them even
declares the arm — adding it is a `required_arms` change, which the checker enforces two-sided.

> **UPDATED 2026-10-10.** No longer true: CPython's two corpora (20 and 32 cases), PostgreSQL's three
> (5, 5 and 9) and SQLite's engine-repros (33) now declare `cheribsd-revocation` and carry measured
> readings; `three-columns-all-programs.md` lists the bundles. Only `sqlite/capi-repros` (19, a
> host-ASan row corpus) still has no CheriBSD arm. The paper sentence quoted above was not re-checked
> here.

**The paper's own "57 counted cases" is itself now stale**: these corpora have grown past it
(memcached 5 -> 11, tshark 12 -> 24, FFmpeg 4 -> 19 rows carrying the arm). That is a number for the
paper's owner to revise, not this document, which reports the tree.

**Corrected below, because this document's cells said otherwise until today** (FFmpeg gained case 3
on 2026-10-03).

| program | counted by the paper | CheriBSD state | evidence |
|---|---:|---|---|
| **Wireshark** | 12 | ✅ **24/24 measured** | ~~13/13~~ **Superseded 2026-10-07.** The corpus is now 22 `wmem` rows plus 2 plain-heap. `wmem-repros/results/20261007-cheribsd-revocation-control` re-ran all 22 at revocation **on** with a **borrowed revocation control that faulted** — the 2026-09-21 bundle had none, only a bounds control and a PoisonCap differential, neither of which shows the revoker sweeps — and on the same image as the other two programs rather than the PoisonCap-tree `3571a6d2…`. `plain-heap-repros/results/20261007-cheribsd`: both rows **CAUGHT**, `si_code` 1, attributed. The 2026-09-21 bundle is kept for its PoisonCap arms |
| **Apache** | 9 | ✅ **9/9 measured** | `apr-pool-repros/results/20260921-cheribsd` (1 case, revocation on *and* off) + `bucket-repros/results/20260922-cheribsd` (8 cases) |
| **memcached** | 5 | ✅ **11/11 measured** | all 5 `case.json` declare `cheribsd-revocation` and their `status` names a CheriBSD run, but `results/` holds no bundle. **Updated 2026-10-04:** the corpus now has a committed capability bundle at `ports/memcached/allocators/results/20261004-qemu-corpus-defects/` (spatial/sublet, 10/10, plus the negative control) and names it in `evidence`, and `e236ca79ea5a` **removed** the `INDEX.md` line this row used to quote. **Superseded 2026-10-07, and one assertion here was flatly wrong by then:** the platform is NOT absent — a stock CheriBSD purecap vehicle was built on this host and the readings were taken on it. `allocator-repros/results/20261006-cheribsd` carries all 8 original rows (0 caught) with a revocation control faulting AND a negative control making all 8 oracles fail, so the completions are not vacuous; `…-case8` carries the 9th, which is **CAUGHT** and refuted its own prediction; `plain-heap-repros/results/20261007-cheribsd` carries 2 rows, one caught and one not. The `results/` ignore that made this the only corpus of sixteen with no committed evidence has been narrowed to raw output |
| **FFmpeg** | ~~3~~ **19** | ✅ **18/19 measured** | ~~4/4~~ **Superseded 2026-10-07:** the corpora now carry 19 rows with the arm across four of them, and 18 are measured — `pool-repros` 4, `subobject-repros` 10, `plain-heap-repros` 4 (3 of them **CAUGHT**), with only `plane-repros/00` unmeasured for want of a purecap `libavutil`. `pool-repros/results/20261006-cheribsd` — all four complete under revocation **on**, with the revocation control FAULTING in the same boot (`tag_after_sweep=0`, then SIGPROT `si_code` 2 at the independently resolved probe). **Updated twice on 2026-10-06:** this cell first read "declared, never claimed" — a false clean from this doc's own `'cheri' in status` probe; it was corrected to "declared *and claimed*, no bundle" when that claim was **retracted** as never run; it is now a real reading. Note the headline **22** above does not include these four: its population is corpora that had a committed bundle when it was written |
| **CPython** | 20 | ❌ **nothing** | no `cheribsd-*` arm in any of the 20 `case.json`; no `status` mentions it; no bundle |
| **PostgreSQL** | 8 | ❌ **nothing** | same three negatives |
| SQLite | 0 (row dashed) | n/a | `required_arms: ["host-asan"]`; no CheriBSD anywhere in that corpus |

So the zero is **measured** for Wireshark, Apache, FFmpeg and memcached — and after 2026-10-07 none
of the four is "in between" any more: each has committed bundles, each on a vehicle whose image hash
is recorded, and each temporal row has a revocation control that faults. It remains **not yet
established** for CPython (20) and PostgreSQL (8), neither of which declares the arm at all.

**And the zero is no longer uniform, which is the sharpest thing this document can now say.** Seven
spatial rows across the three target programs ARE caught. What decides it is measured rather than
argued: the crossing must leave the allocator's **usable size** (CheriBSD's `malloc` bounds to that,
not to the request — refuted two catch predictions at requests 9 and 24, confirmed two at 16 and 64);
a crossing **below the base** is always caught; and **which inner allocator carved the storage**
decides it, memcached's slab-chunk rows completing while its cache-object row faults because the port
malloc's a page for the former and each object for the latter.

### Correction, 2026-10-06: this doc's own probe produced a false clean

The FFmpeg cell said "never claimed" on the strength of the reproduction recipe below, which tests
`'cheri' in str(json.load(open(f)).get('status','')).lower()`. The three `pool-repros` cases *did*
claim a measured CheriBSD run — their `status` read "native pair and **every arm above** run
2026-09-20; all passed", which asserts the arm ran while containing no literal `cheri`. Re-running
the predicate on those three files returns `False, False, False`, so the probe was working exactly
as written and still reported the opposite of the truth.

**This is the keyed-to-one-spelling shape, in an instrument built to audit claims.** A substring
probe over free prose cannot establish an absence of claims; the claim has to be absent from what the
prose *means*, not from one word. A probe that reads `arms[*].status`/`oracle` for a measured/declared
token, rather than grepping the case's summary for a program name, would have caught it.

The claim itself was **retracted** on 2026-10-06 (`bug-corpora/ffmpeg/pool-repros/README.md`, and each
case's `cheribsd-revocation.retraction` field, which is kept rather than deleted).

**And then, later the same day, it was MEASURED** — `pool-repros/results/20261006-cheribsd/`: all four
complete under revocation on, 0 of 4 caught, with the revocation control faulting in the same boot.
So FFmpeg's state did change in substance, twice in one day: *claimed without evidence* → *retracted*
→ *measured*. The predicted outcome was right throughout; what was wrong was asserting it had been
observed before it had.

**RESOLVED, and then overtaken.** This doc counts **22** (Wireshark 13 + Apache 9) and the inventory's
table (c) counted **18** (tshark 13 + memcached 5). That was never a discrepancy — different
populations under different evidence conventions: this doc counts corpora with a *committed bundle*,
which excludes memcached, while the inventory covers only its three programs, which excludes Apache.
The inventory's figure is **now also 22** (tshark 13 + memcached 5 + FFmpeg 4), so the two coincide
numerically while still counting **different sets**. That coincidence must not be read as
corroboration, which is why both compositions are written out here and in the inventory.

## Two numbers that are easy to conflate, and why the distinction matters

A first pass over the `arms` objects gives "60 of 77 cases declare no CheriBSD arm", which **overstates
the problem by a wide margin**:

- ~~**Wireshark declares the arm nowhere and is nonetheless measured 13/13.**~~ **CLOSED 2026-10-04 by
  `fabb59f1f28b`**, which this entry prompted: all 13 cases now declare `cheribsd-revocation` with the
  outcome read from the bundle's own `matrix.tsv`, and the arm is in `required_arms`, so the omission
  cannot recur (verified two-sided: removing it from one case makes `check-corpus.py` exit 1).
  As written: its CheriBSD outcome lived in a committed bundle and in all 13 `status` strings, and only
  the schema's `arms` object omitted it.
- So the **declaration** gap (~~60/77~~ **47/77** after `fabb59f1f28b`) and the **measurement** gap
  (28 firm /57) are different facts.
  Only the second one bears on the paper's sentence.

This is worth stating because the cheap instrument — counting `arms` keys — points at the wrong
conclusion, and the expensive one — opening each bundle — reverses it for 13 cases.

## Why the gate did not catch this

`check-corpus.py` reports **CLEAN** (14 corpora, ~~107~~ **108** declared cases, 0 problems) and
`build-index.py --check` reports **current**. Neither is wrong: the checker validates declarations
against the tree — required fields, numbering, arm well-formedness, liveness proofs — and never asks
whether a declared arm has an **outcome**, nor whether a `status` sentence is backed by a bundle. Every
discrepancy above passes the gate by construction.

~~Across all 77 `case.json`, every Capstone/PoisonCap arm has `status: null`; the only populated
`arm.status` anywhere is `"not written"` on `native-detect`.~~ **Stale since 2026-10-03:** FFmpeg's
case 3 populates `status: "not written"` on `poisoncap-spatial`, `poisoncap-protected`,
`cheribsd-revocation` and `backing` as well, so the claim now holds for 38 `native-detect` arms plus
those four. The point it was making is unchanged. All real measurement lives in free-text
`status` prose, which is why it drifts.

## What this means for the claim

The paper's own completion gate offers two routes, and either settles it:

> "Completion requires the specified evidence **or an explicit narrowing of the corresponding claim**,
> not a favorable outcome." — `appendices/a-evidence-status.tex:7-10`

- **Evidence route:** run the CheriBSD arm for CPython's 20 and PostgreSQL's 8, and commit a bundle for
  memcached's 5. The harness exists — Wireshark's, Apache's and now FFmpeg's bundles are its output,
  and FFmpeg's were produced on a vehicle built on this host, so the platform is no longer a blocker.
- **Narrowing route:** state the denominator the zero was measured over — as of 2026-10-07 that is
  Wireshark, Apache, FFmpeg and memcached and report the rest as not-yet-run, which is what `S3-real-bug-corpus.md:197-203` already
  provides a vocabulary for (`level-not-ported`, `other`).

**Related and more urgent:** the same macros print `\targetCorpus{\targetmeasured{57}}` and
`\targetDetected{\targetmeasured{57}}` as *measured* (`macros/target-results.tex:23-24`), while S3's own
evidence row reads **"Pending. Reproduction campaign outstanding"**
(`appendices/a-evidence-status.tex:135-136`) and `experiments/studies.json` records
`"evidence": "pending"`. There is no `experiments/results/S3/` bundle and no `points.csv` anywhere in
the tree, though the protocol names `points.csv` as the required deliverable
(`S3-real-bug-corpus.md:111-128`). That is a manuscript/artifact inconsistency for the lead to
resolve, not something a lane should paper over by editing either side.

## How to reproduce this table

    cd capstone/bug-corpora
    # per corpus: cases, arms declaring cheri, statuses naming cheri, cheri bundles on disk
    python3 - <<'EOF'
    import json,glob,os
    for c in ['cpython/pymalloc-repros','wireshark/wmem-repros','postgres/mmgr-repros',
              'memcached/allocator-repros','ffmpeg/pool-repros','httpd/apr-pool-repros',
              'httpd/bucket-repros','sqlite/capi-repros']:
        pat = f'{c}/row*/case.json' if 'sqlite' in c else f'{c}/[0-9][0-9]_*/case.json'
        fs = sorted(glob.glob(pat))
        decl = sum(1 for f in fs if any('cheri' in a.lower() for a in (json.load(open(f)).get('arms') or {})))
        st   = sum(1 for f in fs if 'cheri' in str(json.load(open(f)).get('status','')).lower())
        bun  = [os.path.basename(d) for d in glob.glob(f'{c}/results/*') if 'cheri' in os.path.basename(d).lower()]
        print(f'{c:30s} cases={len(fs):3d} declares={decl:3d} status={st:3d} bundles={bun}')
    EOF

Then open each named bundle's `matrix.tsv` and count `arm=cheribsd` rows — **that step is the one that
matters**, because it is what reversed the Wireshark verdict.
