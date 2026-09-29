# Track B, af_join: a pool defect in FFmpeg's real code (capstone-qemu, 2026-09-29)

**Question.** Until now the pool defects ran only as the buffer-pool port's reductions (probe
cases 36–38). This run asks two things:

- Does the af_join defect run as FFmpeg's own code? The defect is the missing reference that
  upstream fixed in `461fb22053`. Running it means a real filter graph, over frames from FFmpeg's
  own decoder pools.
- Does the Sublet port of those pools catch it where stock pools let it read reused memory?

**Answer: yes, 12 of 12 cells as registered, on the final images.** The registration is
`PREDICTIONS.md` (`4a9e6ed46132`) with its three addenda (`150887238a68`, `ee6f030671e1`,
`a2e729d7f93e`). Two problems had to be found and fixed before that run:

- **A compiler defect, now C-69.** It blocked libavfilter entirely.
- **A fixture error.** Against the fixture as first registered, the tally was 9 of 12. Fixture
  19's three poolsublet cells faulted one read early, at the fixture's own bounds print. They are
  recorded as DIFFERS in `result-lines.txt`, section 4.

## The experiment

- **A real graph over pool frames.** Two `pcm_s16le_planar` decoders feed `abuffer` ×2 → `join`
  (`map=0.0-FL|0.0-FR|1.0-FC`) → `abuffersink`. Output channel 1 repeats a buffer and channel 2
  brings a new one, which is the sequence the upstream fix is about.
- **Fixture 18 links `af_join.o` as 9.0.1 ships it.** Fixture 19 links the same file with the fix
  reverted, one token. Both link their object ahead of `libavfilter.a`.
- **The two images differ in af_join's bytes and the fixture id alone.** Three checks establish
  this:
  - Two gates: `make`'s own command, rerun, must reproduce the archive's `af_join.o` byte for
    byte, and so must the route the revert is compiled by. The revert must also be one line.
  - Of the images' named symbols, every one sits either at the same address or exactly 8 bytes
    lower in 19. The reverted `activate` is 8 bytes shorter.
  - An earlier run linked the shipped object from the archive instead, and its pair differed in
    layout too (section 5 of `result-lines.txt`). It is superseded.
- **The missing reference is observed, not only inferred.** Each fixture prints how many
  references the output frame holds: 2 with the fix, 1 without.

## Result (final images, N = 3 per cell)

| | poolstock (FFmpeg's pools as shipped) | poolsublet (the Sublet port of the pools) |
|---|---|---|
| **18, fix present** | RETURN `120005b` ×3; the output holds 2 references | RETURN `120005b` ×3; 2 references |
| **19, fix reverted** | RETURN `1300177` ×3; 1 reference. Input 1's buffer went back to its pool while the output still pointed into it, and the next packet's frame got the same buffer (0x77) | **FAULT temporal ×3** at the touch, on input 1's plane (`0x102c07e00`); 1 reference. The plane was revoked when it went back to its pool |

The bottom-right cell is the protection. The top row shows the port is not faulting on anything
else: the same graph with the fix present completes on both arms.

**The fault is the revoked plane, shown by the tag rather than the address.**

- The Sublet pool reissues the same address, so the value alone cannot tell stale from live.
- The fixture read the new owner's bounds (`lcc`) without faulting. The output's pointer, loaded
  before the next decode, is the untagged one.
- An earlier run faulted on that pointer before any reissue, which dates the untagging to the
  buffer's return to the pool.

The emulator is deterministic: the three reps print identical addresses. So N = 3 shows
repeatability, not three independent samples.

**M1–M5 on both arms, with 0004 applied: 30 frames, 0 hash mismatches, and the flip control
fired.** This shows the rebuilt libraries still decode unchanged. It says nothing about 0004
itself, because the M5 image contains no libavfilter code.

## What had to be found first

**1. A compiler defect, which blocked libavfilter entirely: C-69.** The first run never reached the
defect. Fixture 18 faulted before its touch on both arms, inside `avfilter_graph_config`. It was
bisected in the domain:

- The simplest audio graph (`abuffer` → `abuffersink`) faults the same way. So does level0,
  which never revokes. So the pointer was never freed: its tag was stripped.
- The same graph is clean natively under AddressSanitizer.
- An instrumented `avfiltergraph.c` reads a link's channel-layouts pointer with tag 1 directly,
  and with tag 0 through `FF_FIELD_AT(void *, m->offset, link->incfg)`, with no free in between.
- **The mechanism.** For `*(T *)((char *)&obj + off)`, clang gives the load **alignment 1**. That
  is upstream behaviour (#152575, fork commit `5569bf26f009`), and x86-64 gets it too.
  - **The Capstone defect is in the backend.** It lowers an under-aligned capability access to
    integer bytes: 16 `lbu`, stores to a stack temporary, then an `ldc`. That drops the tag.
  - Stores are lowered the same way, scattered back out as 16 `sb`.

`fieldat-repro.c` reduces it to ten lines. The compiler lane reproduced it independently and filed
it as **C-69**, with a two-sided reproducer at
`capstone/tests/compiler-repros/C69-underaligned-capability-bytecopy/` (branch
`lane/compiler-c69-underaligned-cap`, `3687ff736ba7`).

App patch **0004** works around it for this port, under `__CAPSTONE__` only: `FF_FIELD_AT` states
the pointee's alignment. Two pairs show it working:

- On level0, diagnostic fixture 30 faults without 0004 and returns with it.
- Fixture 18 faulted on both pool arms before 0004, and returned after it, with 0004 the only
  FFmpeg-side change between the two runs.

**2. A fixture error, which moved the fault.** With 0004 applied, fixture 19 on poolsublet faulted
*before* its touch.

- The fixture's own diagnostic print read the bounds of the output's plane, and capstone-qemu
  faults on `lcc` of an untagged operand. The operand was exactly input 1's plane, so the defect
  was caught one read earlier than the registered site.
- The fixture was corrected to read nothing of that plane before the touch, as every other
  fixture here takes its target while the target is live. The prediction did not change.

## What this does not establish

1. **QEMU only** (Q-11): on silicon a stale access retires. The images also depend on
   capstone-qemu fabricating `gp` (every boot logs `gp FABRICATED`), which is QEMU-only as well.
2. **Patch 0004 is a workaround for C-69, in one port.** Other code that reads or writes a pointer
   field through `(T *)((char *)p + runtime offset)` loses tags the same way until the backend is
   fixed.
   - A disassembly scan for the two lowering shapes finds none in the final images or in M5.
   - Its controls fire: it finds 10 sites in the pre-0004 `avfilter_graph_config`, and 1 in each
     reduction.
   - The scan knows only those two shapes, within fixed windows, and it has not been run over the
     other ports' images.
3. **One defect.** vidstab and h264_refs, the plan's other two defects, are not part of this run.
4. **Infrastructure.** Two boots in section 5's run stalled before any section ran. Both guests
   were killed by hand after minutes frozen, so they would not hold the shared QEMU lock for
   16 minutes, and both cells were re-run. Every attempt is kept in `result-lines.txt`.

## Files

- `PREDICTIONS.md`: the registration and its three addenda.
- `result-lines.txt`: every run in order, including the bisection and the superseded runs.
- `fieldat-repro.c`: the compiler reduction.
- `SHA256SUMS`: the final images per arm, section 5's, and the M1–M5 stage images.
