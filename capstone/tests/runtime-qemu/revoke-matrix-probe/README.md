# Revocation enforcement test matrix (cases 2–3)

Extends the M0 `borrow-revoke-uaf-probe` to cover more of the test matrix in
`agent-handoff/design/revocation-enforcement-proposal.md` §6. Same lender flow
(lend a region as `REV_BORROWED`, revoke between two domain calls); the borrower
varies how it holds the delegated capability across the revoke.

- **Case 2 — memory-stored:** the borrowed cap lives in a `.bss` pointer slot,
  reloaded on the round-2 dereference.
- **Case 3 — explicit stc/ldc:** round 1 stores the cap into a separate capability
  slot (stc); round 2 reloads it (ldc) and dereferences.

Both exercise the **cap-load untag** enforcement point (`helper_reg_set_cap_compressed`).

Build/run: `../build-revoke-matrix-probe.sh`, `../run-revoke-matrix-probe.sh`
(both cases run in one guest boot).

## Result (2026-06-30) — dormant gap, as expected
With the enforcement patch committed but the recording side reverted (pending the
author), revocation marks nothing invalid, so both cases show the gap:

```
case 2: round1 0x101; region revoked; round2 0x202; NO-TRAP-GAP store landed
case 3: round1 0x101; region revoked; round2 0x202; NO-TRAP-GAP store landed
```

When the recording fix lands, both must flip to a capability fault; update the
final marker in `../run-revoke-matrix-probe.sh` accordingly.

## Not covered here
- **Case 1 (register-held cap):** covered in spirit by the access-path check in
  `_helper_access_with_cap`; hard to force a register-resident cap across the
  `dom_return` ecall from C, so not a separate probe.
- **Case 4 (senior-cascade sub-cap):** deferred. It needs `SHRINK`/`SPLIT` in the
  borrower to derive a distinct sub-node, but these probes are built with the
  buildroot gcc, which has no Capstone builtins and won't assemble the custom
  instructions. Doing case 4 requires either a Capstone-clang-built borrower or a
  raw-encoded `.insn`. Low priority while the recording side is dormant.
