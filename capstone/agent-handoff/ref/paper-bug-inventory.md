# Paper bug inventory — every specimen, by taxonomy row

*Single index of what we have, what is in flight, and what is still to build. Grouped by the
taxonomy rows of `design/sharing-bug-taxonomy-and-novelty.md`. Updated 2026-08-14.*

## How to read this

**Status:**

| | meaning |
|---|---|
| **BOTH** | Capstone column **and** CHERI column measured |
| **CHERI** | CHERI column measured, Capstone column missing |
| **REPRO** | reproducer exists and runs; no mechanism column yet |
| **TRIAGED** | upstream-verified, not built |
| **BLOCKED** | cannot be built — reason given |

**Compiler feasibility — what our toolchain can actually take:**

* ✅ **Pure C** compiles for `capstone64`. This is the only thing that does.
* ✅ **Rust / Go / JS / C++ bugs are investigable as distilled C shims.** This is the
  established method for all three existing corpora, gated by
  `xlang/cheri/check_shim_fidelity.py`. A bug's *upstream* language does not decide feasibility;
  the *shape* does.
* ❌ **C++ / STL** — no `libc++` for `capstone64`, and a `new`-expression crashes the backend.
  (Excluded sol2 ×2, LuaBridge, wxLua.)
* ❌ **Any JIT** — no `capstone64` backend. (Excluded xmlua, tarantool LuaJIT.)
* ❌ **True concurrency** — single hart, no threads. Anything needing a racing core is out;
  anything expressible via callback re-entrancy is in.

---

## Row 1 — Cross-domain UAF / use-after-close *(the volume; split by lifetime-ender)*

### 1a. Free-ended → the **performance** argument

**Corpus A — SQLite host bindings (`tab:scope`, the paper's corpus).** 15 rows; 11 reproduced
faithfully, 4 modelled (7, 12, 13, 15). CHERI measured on all 15; **Capstone measured on 7.**

| # | Host (lang) | Upstream ref | Class | Capstone | Status |
|---|---|---|---|---|---|
| 1 | CPython (C) | gh-142830 | UAF, progress-handler ctx | — | CHERI |
| 2 | rusqlite (Rust) | **RUSTSEC-2021-0128 = CVE-2021-45713** | UAF, closure lifetimes | `run-sqlite-row2.sh` | **BOTH** |
| 3 | diesel (Rust) | **RUSTSEC-2021-0037** | column-name ptr cached | `run-sqlite-row3.sh`, `-b2` | **BOTH** |
| 4 | PHP (C) | Bug #66550 | use-after-close | — | CHERI |
| 5 | PHP (C) | Bug #69971 | destruction order | `run-sqlite-row5.sh` | **BOTH** |
| 6 | PHP (C) | Bug #77977 | UAF via UDF | — | CHERI |
| 7 | CPython (C) | gh-99886 | cursor dealloc order *(modelled)* | `run-sqlite-row7.sh` | **BOTH** |
| 8 | CPython (C) | gh-85981 / bpo-41815 | backup on closed conn | — | CHERI |
| 9 | sqlite3-ruby | Issue #49 | finalize-time segfault | `run-sqlite-row9.sh` | **BOTH** |
| 10 | sqlite3-ruby | "Closed Statement" | stmt after close | — | CHERI |
| 11 | go-sqlite3 (cgo) | PR #1303 | double-free | `run-sqlite-row11.sh` | **BOTH** |
| 12 | expo-sqlite | Issue #34990 | unfinalized → NPE *(modelled)* | — | CHERI |
| 13 | CPython (C) | gh-149738 | deleted `row_factory` *(modelled)* | — | CHERI |
| 14 | CPython (C) | bpo-31746 / PR #27472 | uninit Connection | `run-sqlite-row14.sh` | **BOTH** |
| 15 | datasette-authz | Issue #3 | authorizer ctx *(modelled)* | — | CHERI |

**TODO:** rows 1, 4, 6, 8, 10, 12, 13, 15 have **no Capstone measurement** — `tab:fix` claims
"All defects are addressed" for all 15. All eight are pure C and feasible; this is the cheapest
outstanding work in the whole paper.

**Corpus B — xlang cross-language FFI.** 15 rows, both columns measured. **9 CVEs + 2
advisories** — the highest CVE density we have, and it is *not* in the paper.

| # | Host | Pair | Ref | Status |
|---|---|---|---|---|
| 1 | rlua 0.8.2 | Lua↔Rust | rlua #19 | BOTH |
| 2 | rlua 0.15.4 | Lua↔Rust | rlua #97 | BOTH *(stack-UAR; never blocked by anything)* |
| 3 | libpulse-binding | Rust→C | **GHSA-f56g-chqp-22m9** | BOTH |
| 4 | mruby | Ruby↔C | **CVE-2022-1071** | BOTH |
| 5 | mruby | Ruby↔C | **CVE-2022-1934** | BOTH |
| 6 | mruby | Ruby↔C | **CVE-2026-1979** | BOTH *(spatial)* |
| 7 | secp256k1 | Rust→C | **RUSTSEC-2022-0070 / GHSA-969w-q74q-9j8v** | BOTH *(cleanest cross-domain lend)* |
| 8 | mruby | Ruby↔C gem | mruby #4926 / **CVE-2020-6838** | BOTH |
| 9 | mruby | Ruby↔C | mruby #3829 | BOTH |
| 10 | mruby | Ruby↔C | **CVE-2022-1106** | BOTH |
| 11 | mruby | Ruby↔C | **CVE-2018-10191** | BOTH *(spatial)* |
| 12 | mruby-io | Ruby↔C gem | mruby #4001 / **CVE-2018-10199** | BOTH |
| 13 | mruby | Ruby↔C gem | mruby #4927 / **CVE-2020-6840** | BOTH |
| 14 | mruby | Ruby↔C | mruby #3596 / **CVE-2017-9527** | BOTH |
| 15 | mruby | Ruby↔C | mruby #3722 | BOTH |

⚠ **Duplication, must be disclosed if used:** rows 8 and 13 are **one upstream defect** (same
vulnerable commit `fc8fb414`, same fix `70e57468`), so 15 rows = **14 distinct fixes**. Rows 4,
5, 8, 10, 13, 15 all share **one shape** (`realloc-vmstack`). 12 of 15 are mruby.

### 1b. GC-ended and allocator-ended → the **security** argument

**Corpus C — Lua-CDP.** 13 measured, both columns, **plus real unmodified Lua**. Zero CVEs.
**7 of 13 are GC-frees** — those are the structural sub-case: the collector reclaims, the
borrower is invisible to it, and no `free()`-triggered sweep can see the link.

| # | Library | Ref | Free direction |
|---|---|---|---|
| 1 | lua-openssl | #141 | native + double-free |
| 2 | ldbus | #20 / PR #21 | native |
| 3 | cffi-lua | #57 | native |
| 4 | lua-SDL2 | #75 / PR #77 | native |
| 5 | Wireshark `wslua` | GitLab #16807 | native |
| 6 | luv (libuv) | PR #696 | GC, cross-thread |
| 7 | lgi | #122 | **GC** |
| 8 | lgi | #65 | **GC** |
| 9 | luaossl | #124 | **GC**, double-free |
| 10 | LuaDBI | #35 | **GC** |
| 11 | luv | #503 / PR #734 | **GC** (coroutine `lua_State`) |
| 12 | Lua-cURLv3 | PR #80 | **GC** |
| 13 | LMDB + binding | *none* — `lmdb.h:249` contract | synthesized |

**BLOCKED (do not retry):** sol2 #1373, sol2 #1080, LuaBridge #319, wxLua #115 — C++/STL.
xmlua #35, tarantool #7657 — LuaJIT. corona-858, tarantool-1955 — not buildable in sandbox.

---

## Row 3 — Reuse-not-free *(no allocator event ever)*

`xlang/reuse-not-free/`. **The paper's motivating example is in this row, not row 1.**

| Specimen | Ref | Evidence | Status |
|---|---|---|---|
| **`sqlite3_column_text` across `step()`** | SQLite docs + `vdbemem.c:301-324` | Built. 4 rows → identical address; **ASan silent, positive control fires** | **REPRO** — Capstone/CHERI columns TODO |
| **`sqlite3_column_name` (lookaside)** | RUSTSEC-2021-0037 shape | Built. **ASan silent with lookaside ON, fires with it OFF** — a *second* blindness mechanism | **REPRO** |
| git `git_path()` ring buffer | patch series, `path.c:30-40` | `strbuf_reset` keeps the allocation → zero allocator events ever | TRIAGED — **easiest, ~20 lines, no deps** |
| Go `database/sql` `RawBytes` | golang/go#65201 | contract + fix commit | TRIAGED |
| LMDB `MDB_val` | `lmdb.h:278` | contract verified; **no filed defect** | TRIAGED |
| cassandra-rs | **RUSTSEC-2024-0017 / CVE-2024-27284** | source-verified single `Row row_;` mutated in place; **advisory misfiles it as CWE-416 "freed memory"** | TRIAGED — hard (needs live Cassandra) |
| curl pool reuse | CVE-2021-22924, CVE-2022-27782 | analogue only — borrower holds no stale pointer | TRIAGED |

**Negative control to keep:** `sqlite3_close()` returns `SQLITE_BUSY` with unfinalized
statements — SQLite defends the invariant everyone else violates.

---

## Row 4 — Hierarchical lifetime violation

Corpus A rows **4, 5, 7, 9, 10, 12** are this row (6 rows, `H` primitive). Additional
specimens found and not yet built:

| Specimen | Ref | Note | Status |
|---|---|---|---|
| Nokogiri `XPathContext` outlives `Document` | GHSA-p67v-3w7g-wjg7 | ~25 lines of libxml2 C | TRIAGED — easy |
| libexpat shared DTD | **CVE-2022-43680** | **Inverted** — the *child*'s free destroys the root's DTD | TRIAGED |
| curl HTTP/2 stream dependency | **CVE-2026-10536** | literal parent/child handle graph | TRIAGED |
| GLib `GMainContext` / `GSource` | glib#803 | child locks the parent's freed mutex | TRIAGED — needs a race |
| Nokogiri `do_xinclude` | GHSA-wfpw-mmfh-qq69 | frees include node + children | TRIAGED |
| sqlite-jdbc | xerial#183 | `sqlite3_finalize` then `sqlite3_reset` | TRIAGED — trivial |
| **curl#17578** | — | **the bridge**: row-4 trigger → row-3 damage | TRIAGED |

---

## Row 6 — TOCTOU / double-fetch *(zero evidence; the gap that matters)*

`xlang/toctou-double-fetch/`. **All callback-expressible — no concurrency needed.**

| # | Specimen | Ref | Severity | Status |
|---|---|---|---|---|
| 1 | **wasmtime bulk `memory.copy`** | GHSA-2hw9-mc66-jc2q / RUSTSEC-2026-0223 | — | TRIAGED — **flagship, build first** |
| 2 | `@nyariv/sandboxjs` | **CVE-2026-25641** | **10.0** | TRIAGED — 10-line conceptual version |
| 3 | Deno resizable ArrayBuffer | **CVE-2023-28445** | **9.9** | TRIAGED |
| 4 | Ladybird cached TypedArray base | GHSA-w89h-j2xg-c457 | **9.6** | TRIAGED |
| 5 | Node.js permission model | **CVE-2024-21896** | High | TRIAGED — patch+PoC in one commit |
| 6 | aio ring `head` | **CVE-2014-0206** | — | TRIAGED — **no race at all**, ~30 lines |
| 7 | BPF ringbuf `consumer_pos` | **CVE-2024-41009** | — | TRIAGED — no race |
| 8 | io_uring SQE re-read | commits `56080b02ed6e`, `9c280f908711` | **no CVE exists** | TRIAGED — 8-line minimal specimen |

**Prior art to cite:** V8 `IterateElements` (CVE-2016-1646 / CVE-2017-5030 / CVE-2021-21225 —
one defect, three CVEs, five years; fixed only by `DisallowJavascriptExecution`), and RLBox
(USENIX Sec 2020) conceding local-variable snapshotting is *"intractable"* and answering with
`freeze()`/`unfreeze()`.

⚠ **Build the architectural shape only** (source reads twice). Compiler-induced ones (Xen
XSA-155/166/197/478) are a codegen lottery.

⚠ **FABRICATED — never cite:** `CVE-2026-50624`, `CVE-2026-50626`, `CVE-2026-61476`,
`CVE-2026-63321`. MITRE returns `CVE_RECORD_DNE`, verified with positive controls.

---

## Rows 5, 7, 9, 10, 11 — supporting / design-only

| Row | Specimens | Status |
|---|---|---|
| 5 Double-free | Corpus A row 11; Corpus C rows 1, 9 | BOTH |
| 7 Callback re-entrancy | Corpus A rows 1, 2, 6, 15 | 1 of 4 measured on Capstone |
| 9 Over-wide / sub-object | Corpus B rows 6, 11 — **both *intra*-domain, not cross-boundary** | BOTH, but weak |
| 10 Uninitialised | Corpus A row 14 | BOTH |
| 11 Provenance forgery | threat-model analysis only | design only |

---

## Summary

| Row | Have (measured both) | In flight | To build |
|---|---|---|---|
| 1a free-ended | 7 (A) + 15 (B) | — | **8 Corpus-A rows, all pure C** |
| 1b GC/alloc-ended | 13 (C), 7 are GC | — | relabel only — no new work |
| **3 reuse-not-free** | 0 | **2 built, host-verified** | Capstone + CHERI columns; git ring buffer |
| **4 hierarchical** | 6 (A, design claim) | — | 7 triaged; sqlite-jdbc trivial |
| **6 TOCTOU** | **0** | — | **8 triaged; build wasmtime shape** |

**Cheapest high-value work, in order:**

1. **Capstone column for the 8 unmeasured Corpus-A rows** — pure C, method established, closes
   the gap between `tab:fix`'s "all 15" and the 7 we actually measured.
2. **Capstone + CHERI columns for `sqlite-column-text`** — expect *both* revoke-on-free and
   every CHERI config to miss it, with only lender-driven revocation catching it. That result
   is the paper's argument in one row.
3. **Build the wasmtime TOCTOU shape** — the only item that changes what the paper *is*.
