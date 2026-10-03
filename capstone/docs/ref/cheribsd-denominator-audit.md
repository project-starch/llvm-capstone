# The CheriBSD denominator: which corpus cases actually carry that arm (2026-10-03)

The nested-allocators paper prints a comparative headline — CheriBSD's deployed heap revocation stops
**0** of the reproduced corpus — and backs it with a sentence that is stronger than a zero:

> "Every comparator now runs the whole denominator, so a zero in `\targetCheriDetected` or
> `\targetSpatialDetected` is a measured miss and never an absent integration."
> — `paper-nested-allocators/appendices/b-target-results.tex:136`

This document is the per-corpus check of that sentence, so the claim can be defended or narrowed from
evidence rather than from memory. **It reports numbers; it does not edit the paper.**

## Answer

**22 of the 57 counted cases carry a committed CheriBSD measurement. 28 carry none of any kind. 8 are
declared-or-claimed with no committed bundle.**

| program | counted by the paper | CheriBSD state | evidence |
|---|---:|---|---|
| **Wireshark** | 12 | ✅ **13/13 measured** | `wmem-repros/results/20260921-cheribsd/matrix.tsv` — 39 data rows over 3 arms, `arm=cheribsd` on 13 distinct cases, all `expected=complete`, `passed=13/13` |
| **Apache** | 9 | ✅ **9/9 measured** | `apr-pool-repros/results/20260921-cheribsd` (1 case, revocation on *and* off) + `bucket-repros/results/20260922-cheribsd` (8 cases) |
| **memcached** | 5 | ⚠️ **declared and claimed, no committed bundle** | all 5 `case.json` declare `cheribsd-revocation` and their `status` names a CheriBSD run, but `results/` holds no bundle; `INDEX.md` already records "commits no result bundle of its own" |
| **FFmpeg** | 3 | ⚠️ **declared, never claimed, no bundle** | all 3 declare the arm; no `status` mentions CheriBSD; no bundle |
| **CPython** | 20 | ❌ **nothing** | no `cheribsd-*` arm in any of the 20 `case.json`; no `status` mentions it; no bundle |
| **PostgreSQL** | 8 | ❌ **nothing** | same three negatives |
| SQLite | 0 (row dashed) | n/a | `required_arms: ["host-asan"]`; no CheriBSD anywhere in that corpus |

So the zero is **measured** for Wireshark and Apache (22 cases), and **not yet established** for
CPython and PostgreSQL (28 cases), with memcached and FFmpeg (8) in between.

## Two numbers that are easy to conflate, and why the distinction matters

A first pass over the `arms` objects gives "60 of 77 cases declare no CheriBSD arm", which **overstates
the problem by a wide margin**:

- **Wireshark declares the arm nowhere and is nonetheless measured 13/13.** Its CheriBSD outcome lives
  in a committed bundle and in all 13 `status` strings; only the schema's `arms` object omits it.
- So the **declaration** gap (60/77) and the **measurement** gap (28 firm /57) are different facts.
  Only the second one bears on the paper's sentence.

This is worth stating because the cheap instrument — counting `arms` keys — points at the wrong
conclusion, and the expensive one — opening each bundle — reverses it for 13 cases.

## Why the gate did not catch this

`check-corpus.py` reports **CLEAN** (14 corpora, 107 declared cases, 0 problems) and
`build-index.py --check` reports **current**. Neither is wrong: the checker validates declarations
against the tree — required fields, numbering, arm well-formedness, liveness proofs — and never asks
whether a declared arm has an **outcome**, nor whether a `status` sentence is backed by a bundle. Every
discrepancy above passes the gate by construction.

Across all 77 `case.json`, every Capstone/PoisonCap arm has `status: null`; the only populated
`arm.status` anywhere is `"not written"` on `native-detect`. All real measurement lives in free-text
`status` prose, which is why it drifts.

## What this means for the claim

The paper's own completion gate offers two routes, and either settles it:

> "Completion requires the specified evidence **or an explicit narrowing of the corresponding claim**,
> not a favorable outcome." — `appendices/a-evidence-status.tex:7-10`

- **Evidence route:** run the CheriBSD arm for CPython's 20 and PostgreSQL's 8, and commit bundles for
  memcached's 5 and FFmpeg's 3. The harness exists — Wireshark's and Apache's bundles are its output.
- **Narrowing route:** state the denominator the zero was measured over (22 cases, Wireshark and
  Apache) and report the rest as not-yet-run, which is what `S3-real-bug-corpus.md:197-203` already
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
