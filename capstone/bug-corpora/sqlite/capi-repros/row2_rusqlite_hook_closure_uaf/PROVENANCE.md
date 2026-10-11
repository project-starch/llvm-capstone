# PROVENANCE — row2 (rusqlite RUSTSEC-2021-0128 / CVE-2021-45713)

- Upstream : https://rustsec.org/advisories/RUSTSEC-2021-0128.html
- Issue    : https://github.com/rusqlite/rusqlite/issues/1048
- Affected : rusqlite 0.25.0-0.25.3, 0.26.0-0.26.1. Patched: 0.25.4, 0.26.2.
- Tier     : **LITERAL-traceable**

**Real memory-safety bug: YES** (use-after-free). Non-`'static` closure bound let a
callback borrow a stack/heap value dropped while the callback stayed registered.
Full verbatim reproducer (the issue #1048 `update_hook` PoC), the real
`call_boxed_closure` trampoline, and the `+ Send + 'static` fix bound are quoted in
the repro header: `before-faithful.c`. See also
`design/sqlite-corpus-fidelity-ceiling.md` §Provenance tiers.
