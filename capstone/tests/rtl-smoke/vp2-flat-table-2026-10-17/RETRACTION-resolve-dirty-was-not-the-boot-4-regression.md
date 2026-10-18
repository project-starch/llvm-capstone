# Retraction: `resolve_dirty`'s default was NOT the boot-4 regression

## What is withdrawn

`8bc0c4dd88cc`'s commit message and
`RESULT-a-qemu-34-of-41-does-not-license-a-fault-path-default.md` (`695fdc90e774`) both attribute
boot 4's regression to `resolve_dirty` defaulting on. **Withdrawn.**

Boot 4b ran `a00109e9`, which is the same module with that default reverted — it read back
`RESOLVE_DIRTY:N:default` — and it **fails identically**: `vm-contract`'s `ok` and `stale` give
`cause=24 pc=…1d94 address=0x2 entry=unknown`, retval 922, and every suite step after is 139 with
the same line.

I reached that attribution from a source diff by elimination — three changes, two apparently inert,
so the third must be it — and published it as "the default was the whole variable" without a boot
that isolated it. Elimination across changes I had not individually tested is not an attribution.

## What survives, and is now better evidenced

**The module is still the variable.** The board lane compared stage sums by script: only
`capstone_vm.ko` and the suite differ from r13, and the failing rung (`vm-contract`) uses no suite.

**And module CONTENT is the variable, not build noise** — a null-change control I should have run
before attributing anything: rebuilding `a56fc415`'s exact source gives a **byte-identical** `.ko`
(`a56fc415ad7133aa`). The build is reproducible, so a bisection over content is meaningful.

**Section layouts are ordinary**: `.text` +4 bytes, one new `__param` entry, string and symbol
tables grown. No new section, nothing structurally odd.

**The escape-print code is provably inert.** `if (diag_escapes && escape_prints < CV_ESCAPE_BUDGET)`
short-circuits, so with the parameter false neither the counter nor any `t->frame[...]` is read.
Nothing on the escape path touches a frame word or takes a lock before that check. The only
*unconditional* additions anywhere in the diff are three `WRITE_ONCE` on a static bool, and those
belong to the Q-26 change.

So by elimination the suspect is the **Q-26 wait rewrite** — and this time that is being settled by
a boot, not asserted. Two single-change variants of `a56fc415` are staged: `bisect-q26`
(`69c11f908ad27c6b`, banner tag `bisect-q26only`) and `bisect-esc` (`a2a0b2f49d790cc6`, tag
`bisect-esconly`).

**Pre-registered:** `esconly` passes, `q26only` fails. The row that matters is the other one — if
`esconly` FAILS, the regression is not logic at all but something about content, size or layout, and
no logic-level reasoning about this module can be trusted until that is explained.

## The instrument finding is unaffected — in fact it is now stronger

The companion result's *thesis* stands; only its attribution was wrong. The numbers, all on the
reference pin with `svadu=false`:

| module | QEMU | silicon |
|---|---|---|
| `a56fc415` | 34/41, seven false | **passes** (r13 identity) |
| `29c8e54e` (`resolve_dirty` default on) | **34/41, seven false** | fails every context |
| `a00109e9` (default off) | **34/41, seven false**, 0 BUGs, 0 WARNINGs | fails every context |

**Three modules, one QEMU reading.** Two of them break every virtual context on silicon, including
Tier 1. So the correct statement is broader than the one I published: it is not that a 34/41 fails
to license a *fault-path default*, it is that **this suite's 34/41 is insensitive to whatever class
of module change this is**, and cannot be used to clear a module for a board record at all. The
faulting instruction supports that reading — `memset+0x2c` with a destination cursor of 2 is content
corruption reached before HELLO, which QEMU's page states never produce on these paths.
