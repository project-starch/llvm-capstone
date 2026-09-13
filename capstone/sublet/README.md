# Sublet: the primitives every port calls

`sublet.h` is the discipline, expressed as operations on capability slots. It knows no
program. A port of an allocator is the patch that calls these operations at that allocator's
five operations, and it lives with the program, in `ports/<program>/sublet/`.

| | |
|---|---|
| here | `sublet.h`, the primitives: split, take, take_linear, handle, give, give_to, move, store, carve, clear, and the three readers |
| `ports/<program>/sublet/` | the patch that applies them to one program's allocators, and the bookkeeping the paper counts |

A linear capability never sits in a C variable. Every primitive takes the address of a slot
and leaves the slot holding what the operation produced, so a copy the compiler makes cannot
move one away.

## Why the primitives are not in a port

The first port put them beside SQLite's patch, because there was one port. From the second
they are shared: the recipe is the claim (`A7`, H3, "the recipe is the same for all five"),
and a recipe that exists twice is two recipes. Moving them was the layout's own rule, written
down in `docs/design/repo-layout.md` before the second port began.

## How a build gets them

Only when a Sublet patch is applied. `build-sqlite-capstone.sh` and `build-sqlite-silicon.sh`
put this directory and the patch's own directory on the include path inside the branch that
applies the patch, and nowhere else, so the unprotected arm of every measurement is a build
that never read a file from here. That arm is what the overhead is measured against.

## Changing a primitive

Every port calls these, so a change reaches all of them. The operations are fixed by the
architecture (opcode `0x5b`: revoke, lcc, split, mrev, init), and the file's header comment
records what each one costs and which of them writes the block through before `init`. A new
operation in a port's patch that is not here means the recipe has grown a case, which is
A7's H3 disproved, and it belongs here with the reason.
