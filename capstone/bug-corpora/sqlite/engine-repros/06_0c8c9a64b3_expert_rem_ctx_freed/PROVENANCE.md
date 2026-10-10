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
