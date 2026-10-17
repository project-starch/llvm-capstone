# expertrem -- 0c8c9a64b3

Upstream fix `0c8c9a64b3`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row19 / sqlite-0c8c9a64b3 -- sqlite3expert idxPopulateStat1() registers rem()/
 * sample() on the USER connection (default iSample=100 -> dbrem = p->db), with a
 * heap IdxRemCtx (rem) and a STACK IdxSampleCtx (sample), then frees the ctx and
 * returns WITHOUT unregistering (sqlite3expert.c:1671/1676/1719). The functions
 * stay registered pointing at freed memory; calling rem() runs idxRemFunc over the
 * freed context. Fixed 3.46.1. CONTROL: run the expert flow, then SELECT rem(...).
 * On unprotected capstone the freed-context read completes -> NOTRAP.
 * Build in the ext group: DOMAIN_EXTRA_SRC includes ext/expert/sqlite3expert.c and
 * -I points at ext/expert for this header.
```

## One arm rejects it; the others differ only in whether an assert is compiled in (2026-10-15)

The defect and the Sublet verdict stand. Probe site 21 sits at the first load from the freed
context -- `assert( iSlot<=p->nSlot )` at `sqlite3expert.c:1429`, `nSlot` being offset 0 of
`IdxRemCtx`, which is where host ASan reports the use-after-free -- and reports both what the
load does and what it returns.

| configuration | at the stale load | afterwards |
|---|---|---|
| host ASan | reports heap-use-after-free, 0 bytes inside a 128-byte region | — |
| virtual-malloc | succeeds; `kind=UAF off=0 n=4`, `nSlot=-1` | the extension's own assert fails, SIGABRT |
| virtual-nested-pools | **rejected**, cause 25 in `idxRemFunc` | — |
| CheriBSD purecap | succeeds; `tag=1 cb=<arena>+1048576`, `nSlot=14400` | runs to `RETURNED rc=0` |

**Only Sublet rejects the invalid access.** The base arm's SIGABRT is the extension's own
`assert`, not a mechanism: the load completed, and what stopped the program was garbage failing a
consistency check the program wrote for itself. The judge reads that as
`NO-READING:inconclusive`, "reached, then neither completed nor faulted", which is the honest
reading -- but it is worth saying what the inconclusive is. On CheriBSD the probe also shows why
nothing rejects the load: the pointer's bounds are the whole 1 MiB memsys5 arena this case
configures, and the access is inside them.

**The arms do not compile the same program.** `ports/sqlite/cheribsd/build-corpus-cheri.sh` puts
`-DNDEBUG` in `SQLITE_DEFINES`, and that reaches the extension's translation unit;
`repro322/build-virtual.py` defines it nowhere, and `sqlite3.c`'s own internal `#define NDEBUG`
does not cross into a separately compiled extension. So the assert is live on the virtual arms
and compiled out on CheriBSD, and the two do not even reach the same first access: with the
assert live it is line 1429, without it the switch below. Case 22 (`spellfix`) is the other
extension case and has the same asymmetry.

**The garbage is not stable.** `nSlot` read back as -1 natively and on virtual Capstone, and
14400 on CheriBSD, where `iSlot=0 <= 14400` would have satisfied the assert even had it been
compiled in. Whether the base arm aborts therefore depends on what happened to be left in the
freed block, which is another reason it is not a detection.

Probing this case needed the injector to reach a file other than the amalgamation; `shared/probe/`
gained that, and the probe's output helpers are no longer static so a site in an extension's own
translation unit can use them.
