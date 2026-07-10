# SQLite corpus — fidelity ceiling and the two residuals

*2026-07-10. Decision record for how faithful the SQLite defect corpus can be
made, which rows can reach a LITERAL real-SQLite matched pair, which cannot and
why, and what the two documented residuals (U, S) would need to fully resolve.
Companion to `benchmarks/sqlite/cve-repros/stage2-mapping.md` (the table) and
`history/10-07-2026_16-30-00_faithful-matched-pairs-per-shape.md` (task-010, the
5 literal reps).*

## Two fidelity axes, scored independently

A matched pair is "same program, two outcomes." The two halves are scored apart:

- **Host "before" (real SQLite + ASan).** A-lane, no QEMU.
  `cve-repros/run-host-asan-repros.sh`. Two sets:
  - essence `before.c`: **18/18** reproduce their `oracle`.
  - binding-faithful `before-faithful.c` (models the real binding's C glue):
    **6/6** reproduce their crash class. The faithful crash *class* can differ
    from the essence one, so it is scored against an optional per-row
    `oracle-faithful` (falls back to `oracle`). Row 8 is the case that forced
    this: the essence fabricates a UAF, but CPython's real path nulls `self->db`
    on close and passes NULL to `sqlite3_backup_init` → **null-deref**, not UAF.
- **Capstone "after" (real SQLite in a domain → capability fault).** B-lane, QEMU.
  **5 LITERAL** (rows 3/R, 11/L, 14/U, 7/H, 2/S — one per shape), **12 mechanism
  probes**, 2 N/A. LITERAL = real SQLite linked, the faulting handle is the one
  SQLite's own API returned/holds, revoke fires on the real lifecycle event.

## Which of the 12 probe rows can become LITERAL

Not uniform — and the split is itself a finding: **a capability lifetime can only
enforce a real pointer-lifetime event.** Rows whose "bug" is not a pointer
lifetime event have no lifecycle hook to hang a fault on, so LITERAL is not
meaningful for them; the probe is the honest ceiling.

| Rows | Shape | Literal-convertible? | Why |
|---|---|---|---|
| 4, 5, 9, 10 | HIER-REV | **YES — clean cascade** | real UAF on a child statement after the parent connection/wrapper is closed; the hier sub-arena substrate (`revoke_on_free_hier_alloc.h`) already exists and row 7 proves the mechanism. Host-faithful halves verified (before-faithful 4/5/9/10 all reproduce). |
| 8, 12 | HIER-REV | YES, but null-deref flavor | not a clean child-UAF: the real bug is use-after-close resolving to a NULL/closed handle (row 8 backup source = NULL; row 12 close path). Convertible, but the fault path is closed-handle/UNINIT-ish, not a plain revoke-UAF. |
| 1, 6, 16 | SEALED-CB | YES — via the S trampoline | free riders on resolving residual S (below): same callback-context mechanism as row 2. |
| 13 | BORROW-REV | **NO — host-language event** | the lifecycle event is a *Python* `row_factory` attribute deletion, "no SQLite C call." There is no SQLite API event to revoke on; imposing one would be *less* faithful. |
| 18, 19 | BORROW-REV | **NO — stale-state, memory-safe** | memory-safe on the host (SQLite reuses the borrow storage in place — no free/revoke event to hook; row 19 was skipped in task-010 for exactly this). The bug is logic/stale-state, not memory safety. |

**Realistic LITERAL ceiling ≈ 14/17, not 17/17:** L + U + all of H + all of S +
R's row 3. Rows 13/18/19 stay probes *by nature*; stating that is a stronger
claim than pretending everything converts.

## Residual U — the UNINIT connection (row 14)

**What's minted vs faithful.** The fault is on a genuine UNINIT capability, but
`sqlite3_open(file, &db)` does not initialise a caller-provided region in place —
it mallocs a fresh `sqlite3` and writes the *pointer* into `&db`. SQLite never
hands the domain an uninitialised-but-typed region, so the UNINIT `db` is *minted*
to model the uninitialised connection rather than *arising from* SQLite's own
allocation.

**What a fully-literal U would need.** The revoke-on-free allocator already routes
all of SQLite's `malloc`, so the `sqlite3` struct is carved from our arena. We
could hand the connection object's allocation back as UNINIT and let
`sqlite3_open`'s initialising **stores** transition it — except `sqlite3_open`
**reads back fields mid-construction**, and loads through UNINIT fault (cause 26)
before init finishes. A literal U therefore needs a semantic refinement:
*the owner domain may load bytes it has already stored into an UNINIT region*
("owner reload"). That is a real, small emulator change, not intra-domain
composition.

**Decision: do NOT force U in code now.** The minted form faults on a real UNINIT
cap and the write-up is accurate. Record "owner-reload UNINIT" as a candidate
primitive; revisit only if the paper leans on UNINIT beyond row 14. Low ROI.

## Residual S — the seal proper (row 2 and the SEALED-CALLBACK family)

**What is already literal.** Real SQLite drives the UDF through its function
table; the revoked `pApp` context faults (cause 24). Faithful — as far as it goes.

**What is missing.** A *sealed* capability is Capstone's cross-domain primitive: a
cap usable only by crossing into its owning domain via `__domcall`/`__seal`. The
faithful sealed-callback story runs the callback in a **separate protection
domain** that SQLite's dispatch crosses into via a sealed entry, `pApp` unsealed
only inside. Our repro runs the callback in the driver's own domain — so no
crossing happens; what we exercise is BORROW-REVOKE on `pApp`, not the seal.

**What a fully-literal S needs.** SQLite calls the callback through a plain C
function pointer. Two paths:
- Heavy: recompile SQLite so indirect calls to registered callbacks emit
  `__domcall` through a sealed entry (callback boundary = domain boundary).
- **Light and sufficient — a callback-domain trampoline:** register a trampoline
  as the UDF; SQLite calls it normally; the trampoline `__domcall`s into a small
  callback domain holding `pApp` sealed; revoking `pApp` (or tearing the callback
  domain down) makes the **sealed call itself** fault. Real SQLite drives it, the
  seal is actually executed, and all four S rows (1/2/6/16) upgrade at once.

**Decision: resolve S — it is the highest-value work left.** It is the only shape
where we hold merely the degenerate (same-domain) form, and the only artifact
that exercises Capstone's *compartmentalisation* (not just revocation) against a
real bug. Delegated to B (task-011). If the paper claims anything about sealing,
this is the gap a reviewer presses hardest.

## Provenance tiers for the faithful host halves

A `before-faithful.c` should be **traceable to real upstream artifacts**, not a
paraphrase of the advisory. Each carries a `PROVENANCE` block citing the exact
issue/PR/commit and quoting the real reproducer + fix hunk. Grounding in primary
source is also a *filter*: it exposes which rows are backed by a real
memory-safety PoC and which are our own constructions. Three tiers:

- **LITERAL-traceable** — verbatim reproducer + real fix hunk from the upstream
  bug report, lowered to C. Rows verified so far:
  - **row 2** (rusqlite RUSTSEC-2021-0128 / CVE-2021-45713): lowers the *actual*
    issue #1048 reproducer — a non-`move` closure borrowing an `Arc<Mutex<()>>`
    registered via `update_hook`, the Arc dropped at scope end, an `INSERT` firing
    the still-registered hook. Real glue quoted: `create_scalar_function`'s
    `+ Send + 'static` fix bound and the `call_boxed_closure` trampoline
    (`sqlite3_user_data(ctx).cast::<F>()` deref) from `src/functions.rs`.
  - **row 8** (CPython bpo-41815 / gh-85981): quotes the verbatim regression test
    `test_bad_source_closed_connection` (Lib/sqlite3/test/backup.py) and the exact
    fix hunk `if (!pysqlite_check_thread(self) || !pysqlite_check_connection(self))
    return NULL;` (GH-22322) added to the backup method in
    Modules/_sqlite/connection.c.
- **MODEL (not traceable)** — a plausible, mechanism-faithful model of the shape,
  but NOT a lowering of a documented CVE. Must be labeled as such, never dressed
  up with false provenance.
  - **row 16** (datasette-sqlite-authorizer): the cited upstream link (issue #3)
    is a **functional test-failure** report (read-only-protection tests failing on
    Python 3.11) — **no UAF reproducer, no vulnerable source, no fix commit**. So
    row 16's "authorizer context UAF" is a *constructed* essence of the
    SEALED-CALLBACK shape. Header labels it MODEL, not LITERAL.

**Action item (do more of this):** sweep the remaining rows' upstream URLs and
either (a) ground the faithful half in the real reproducer + fix, or (b) demote to
MODEL with an honest label where no memory-safety PoC exists. Each such pass
either strengthens a row or surfaces an honest gap — both are wins for the table.

## Priority

1. **S seal-proper trampoline** (B) — resolves residual S, lifts rows 1/2/6/16.
2. **H family 4/5/9/10 → literal** (B, follow-up) on the existing hier substrate;
   host-faithful halves already verified A-side.
3. **U residual** — design note only (this doc); no code now.
4. **13/18/19** — deliberately stay probes; scope result, not a gap to close.
