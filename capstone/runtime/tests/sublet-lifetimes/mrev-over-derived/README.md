# MREV over a node that already has CDERIVE children

This probe backs the LINEAR-duplicates paragraph of `ISSUES.md` Q-15. It runs on capstone-qemu's virtual
profile.

MREV takes a LINEAR operand and CDERIVE a NONLIN one. A handle over a node with derived children
therefore needs two copies of the node at once. One copy stays LINEAR for MREV; the other is
delinearised and derived from. Without a duplicate this cannot happen: a handle over a node is taken
while a LINEAR reference still names it, before any child exists. Case 23 of the QEMU submodule's
`tests/sublet-lifetimes` gate uses that ordering, and it is by design.

capstone-qemu supplies the duplicate: SHRINKTO with rd ≠ rs1 leaves its LINEAR source in place (R-21's
box, where the emulator's arm 10 reads 0). `program.inc` does:

```
SHRINKTO a4, s4, 1024    s4 stays LINEAR
DELIN    a4              a NONLIN copy of s4's node
CDERIVE  a5, a4, 64      a child of that node
MREV     s9, s4          succeeds; a5 still loads
REVOKE   s9              a5's next load faults with 25
```

`s4` is minted RW without MANAGE, and the probe checks this. Neither MREV nor legacy REVOKE reads a
permission, and case 23's handle has no MANAGE either. What the duplicate changes is when the handle can
be taken.

| arm | what changes | expected |
|---|---|---|
| `dup` | the sequence above | cause 25 at the load after REVOKE |
| `norevoke` | no REVOKE | cause 3 at the `ebreak` |
| `inplace` | SHRINKTO in place, then `MOVC a4, s4` | cause 26 at the MREV, whose operand MOVC has nulled (QEMU's cause for an untagged MREV operand) |
| `wrongpc`, `wrongcause`, `wrongnodes` | `dup` with a wrong fault PC, cause or node count | must FAIL |

Every arm also checks the exact fault PC and the exact number of allocated nodes. A mutation that expects
MANAGE on `s4` fails `dup`, `norevoke` and `inplace`.

**Result, 2026-10-13: 12/12** on flat and paged node tables. QEMU was a build whose `target/` and `hw/`
equal capstone-qemu `7dbfdd534eea` (sha256 `a08a0e37b041d6ac`). The clang was
llvm-capstone `bd372e25f5fb`, used only as the host assembler and linker.

Run:

```
bash capstone/runtime/tests/sublet-lifetimes/mrev-over-derived/run.sh
```

It needs the QEMU submodule's `tests/` and a built binary. Override them with `CAPSTONE_QEMU_SRC` and
`VIRTUAL_CAPSTONE_QEMU_BINARY`. It also needs a host `clang` with `ld.lld` on `PATH`. The binary's
sha256 and the clang used are written to `identity.txt` in the log directory.

This is a probe, not a regression gate. `dup` passes only while SHRINKTO copies. Once capstone-qemu
moves the LINEAR source, as the specification requires, `dup` stops before any handle exists. That
failure is what the fix should do, not a regression.
