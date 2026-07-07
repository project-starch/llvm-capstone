# SQLite sealed-callback context-revoke probe (callback UAF)

The Stage-2 **"after"** for the SEALED-CALLBACK class (cve-repros rows 1/2/6/16:
cpython progress-handler UAF, rusqlite hook-closure UAF, php UDF UAF, datasette
authorizer-context UAF, primitive **S** with L/R/H). A host registers a callback
whose context pointer (SQLite's `pApp`) is later freed/replaced while the engine
still holds the registration; a subsequent invocation dereferences the stale
context — a use-after-free.

## Result (2026-07-07) — TRAPPED (composes from existing ops)

```
sqlite-sealed-cb: callback registered (context borrowed to engine)
sqlite-sealed-cb: round 1 (invoke while registered) retval = 0xca11bac0ca11bac0
sqlite-sealed-cb: engine read callback context OK while registered
sqlite-sealed-cb: callback unregistered (context revoked)
sqlite-sealed-cb: round 2 returned 0x000000000fa017ed
sqlite-sealed-cb: callback use-after-free TRAPPED (sealed invocation faulted on revoked context)
```

`round 2 == 0x0FA017ED` (fault sentinel) == the engine's sealed re-invocation of the
callback faulted on the revoked context; the monitor terminated the domain.

**Feasibility verdict:** the SEALED-CALLBACK shape needs **no new firmware op**. It
composes from ops already validated: `shared_region_annotated(PERM_IN, REV_BORROWED)`
+ `revoke_region` (the BORROW-REVOKE mechanism, applied to the callback *context*)
plus the fact that a domain is itself a **sealed** capability — the monitor's
`create_dom` builds it with `__seal`, and every `call_dom` is therefore a sealed
invocation of the callback entry. So Step 2 (a dedicated sealed-callback monitor op)
was **not** required.

## Mapping onto the monitor-mediated model

| Role | Harness side | SQLite meaning |
|---|---|---|
| host binding | lender / `.user` | owns the callback context (`pApp`); registers, invokes, unregisters |
| engine | callee / `.smode` (sealed domain) | on each sealed invocation runs the callback body, which stashed `pApp` and reads it |
| register | `shared_region_annotated(PERM_IN, REV_BORROWED)` | `sqlite3_progress_handler` / `sqlite3_create_function` / `sqlite3_set_authorizer` |
| unregister / replace / close | `revoke_region` | `set_authorizer(db,NULL,NULL)`, UDF replacement, handler removal, connection close |

## Flow

- The host lends the callback context as a `REV_BORROWED` region ("register").
- Round 1: the engine invokes the sealed callback, which caches `pApp` in a `.bss`
  slot (as a real callback stashes its registration context) and reads it
  (`0xCA11BAC0…`).
- The host `revoke_region(context)` == unregister / replace / close (frees `pApp`).
- Round 2: the engine invokes the callback again and re-reads the cached `pApp` =
  the use-after-free → the cached capability reloads untagged → the read faults.

## What this proves — and what it does not

- **Proves (buggy-host model):** a callback-context UAF becomes a deterministic
  fault when the context is revoked at unregister/replace/close. This is what the
  Stage-1 repros for rows 1/2/6/16 reproduce.
- **Rests on argument, not separately probed (malicious-host model):** that a buggy
  or hostile host cannot *forge* a callback entry or invoke a stale sealed one —
  this follows from the domain being a `__seal`-ed capability (unforgeable,
  provenance-carried) rather than from a distinct probe here.
- **Honest framing:** structurally this is the validated BORROW-REVOKE mechanism
  applied to the callback *context*, with the sealed domain entry as the invocation
  vehicle. The "S" in the row primitive is the *unforgeability/invocation* property
  of that entry, supplied by the existing domain seal.

Build/run: `../build-sqlite-sealed-callback-revoke-probe.sh`,
`../run-sqlite-sealed-callback-revoke-probe.sh`. Runs against the current firmware
(existing ops only — no rebuild). Sibling shapes: `sqlite-borrow-revoke-probe` (R),
`sqlite-hier-child-revoke-probe` (H).
