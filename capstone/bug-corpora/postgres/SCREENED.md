# Candidates examined and not collected

The backport sweep of `REL_17_5..REL_17_STABLE` (859 commits, the complete set
of fixes backpatched into 17 after 17.5 — verified as `60332 - 59473 = 859`)
produced more candidates than cases. This file records the ones that were read
and rejected, with the reason, so that the candidate count is not mistaken for
a yield and so that none of them is re-examined from scratch.

**Rejected because the defect is not memory safety**

| commit | subject | why not |
|---|---|---|
| `7a522039f7` | Fix mb2wchar functions on short input (CVE-2026-2006) | upstream states it outright: "While it **didn't overrun the buffer**, it was surely garbage output." A wrong-output bug |
| `0c9cbbfb5b` | Fix off-by-one with NFC recomposition for Hangul U+11A7 | TBASE is wrongly treated as a valid T syllable, so a character is "silently swallowed". Incorrect result, no bad access |
| `fe0b5bd6de` | Harden tsvector code against overflows | upstream: the field overflows "couldn't do anything much worse than produce a corrupted tsvector value". The downstream integer overflows it warns about are not demonstrated to reach a bad access |
| `1cd783d205` | In pg_dumpall, don't skip role GRANTs with dangling grantor | "dangling" here is a dangling catalog reference, not a dangling pointer |
| `3f10d2b665` | Fix PQport to never return NULL unless the connection is NULL | NULL dereference in the caller. A validity bug, not a bounds or lifetime one |
| `2805e1c1ed` | ecpg: Fix NULL pointer dereference during connection lookup | NULL dereference |
| `5d67549d94` | Add missing connection validation in ECPG | NULL dereference in four ECPG entry points |
| `4c5485aedf` | psql: Fix psql slash option leaks | memory leak |
| `351e59f344` | Avoid memory leak on error while parsing pg_stat_statements dump file | memory leak |
| `e20b3256ae` | Avoid resource leaks when a dblink connection fails | resource leak |
| `d07bc7c2b3` | Fix dumping of comments on invalid constraints on domains | dump ordering |
| `6b755d8d70` | pg_dump: include comments on not-null constraints on domains | dump completeness |
| `92268b35d0` | pg_dump: Fix compression API errorhandling | error-handling refactor |
| `dca6627de0` | Save/restore more lexer state when skipping text due to \if | lexer state, no bad access. Prerequisite for CVE-2026-6464's fix, which is itself not a memory defect |

**Rejected because the defect cannot fire on a 64-bit arm**

All three arms are 64-bit (Capstone riscv64, CheriBSD riscv64-purecap), so a
`size_t` wraparound that needs a 32-bit `size_t` is unreachable for us. These
are real defects upstream and are listed so that the exclusion is a recorded
decision rather than an oversight.

| commit | subject | why not |
|---|---|---|
| `87357a606e` | Avoid overflow in size calculations in formatting.c | upstream: "This is **harmless on 64-bit systems** where we'd compute a size exceeding MaxAllocSize and then fail, but on 32-bit systems we could overflow size_t" |
| `f5999f0181` | libpq: Prevent some overflows of int/size_t | the fix widens `int` to `size_t` and adds overflow guards. On 64-bit the wire-supplied lengths are bounded by `int`, so `clen + 1` converted to `size_t` becomes a huge value and the allocation fails rather than being undersized. **Not conclusively established** — if a 64-bit path is found this moves into the corpus |

**Rejected because the trigger is not deterministic**

| commit | subject | why not |
|---|---|---|
| `52b3e7001c` | pgbench: fix verbose error message corruption with multiple threads | a genuine use-after-free on a `malloc` object, and the only one found in the whole window: `printVerboseErrorMessages` keeps a function-local `static PQExpBuffer`, so one thread's `enlargePQExpBuffer` frees a block another thread still holds. **It is reachable only through a data race**, and all three arms are single-threaded domains. This is the sole candidate for the empty `temporal, non-nested` cell |

**Collected but not yet built**

| commit | subject | bucket |
|---|---|---|
| `c4d04cc481` | Guard against overflow in "left" fields of query_int and ltxtquery | spatial, nested |
| `838248b1bf` | Fix encoding length for EUC_CN | spatial, non-nested — but upstream calls the overrun "hypothesized", so it needs a demonstrated trigger before it is a case |
| `e3a2bea41c` | Harden our regex engine against integer overflow in size calculations | spatial, nested |
| `26dd3cac20` | Fix integer-overflow and alignment hazards in locale-related code | spatial, nested |
| `c97a286185` | Guard against overly-long numeric formatting symbols from locale | spatial, nested — needs locale control |
| `ea5f0d176a`, `a5426dbf84` | Prevent buffer overruns in spell.c's parsing of affix files / CheckAffix() | spatial, nested — needs dictionary files in the fixture |
| `3c41f5534a` | Fix integer overflow in array_agg() | spatial, nested — needs a very large array |
| `8e87bd4731` | Harden tsquery code against overflows | spatial, nested — upstream calls it theoretical and it needs a tsquery_rewrite expansion large enough to overflow an int |

**Already in the corpus, identified by this sweep**

The sweep also supplied the upstream fix for three cases that had been carrying
`no-fix-recorded` or an advisory id alone:

| case | commit | how it was matched |
|---|---|---|
| `backend-175/02` ts_headline | `3ed3dbbf44` (CVE-2026-6473) | upstream fixes an `int16` option length; the case's trigger is `repeat('x', 32768)`, which is `PG_INT16_MAX + 1` exactly |
| `backend-175/03` ltree lquery | `8c34261109` | upstream fixes a `uint16` `totallen`; the case's trigger is 66 variants of 1000 characters, about 66 KB against a 65535 ceiling |
| `client-repros/01` pg_dump transforms | `c2b16f5d49` (Security: CVE-2026-19385) | upstream describes the same two facts the case rests on, including "if there are exactly FUNC_MAX_ARGS OIDs, then parseOidArray won't zero-fill any entries, allowing the subsequent loop to run off the end" |
