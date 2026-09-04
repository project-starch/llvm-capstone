# F-04: csmith seed 7 wedges the guest at -O0 and matches native at -O2

**Found by the csmith differential campaign on 2026-09-05 (cycle-2 compiler, QEMU
5dc356547d7f).** Not root-caused.

| build | result |
|---|---|
| native x86 -O0 (the reference) | checksum 0x1E21A964 (505522532) |
| capstone64 -O2 (`cs7-O2.dom`, sha256 21498ff73a7661a6) | RET 505522532 -- MATCH |
| capstone64 -O0 (`cs7-O0.dom`, sha256 d1f543e17b0f8da7) | WEDGE: the shell prompt never came back; QEMU stayed alive |

The only guest output before the wedge is QEMU's `[CAPSTONE] Print = Scalar(0x1234)`:
`helper_csdebugprint`, the debug-print instruction (funct7 1000011 under OPC_CAP_OP), which
the compiler never emits and which does not occur in the image (`llvm-objdump -d` shows no
`<unknown>` encoding). So at -O0 control flow LEFT the code and executed bytes that decode as
that instruction. The program has no `switch`, no indirect call, no indirect jump, and its
whole stack demand is under 3 KB (largest frame 704 bytes, 21 frames, 2944 bytes in total),
so neither a jump table nor a stack overflow explains it. Reproduced twice (the first
campaign, before the batch runner learned to reboot after a wedge, and this one).

Reproduce (build then run; `capstone/tests/fuzz/build-fuzz-program.sh`, then
`capstone/tests/fuzz/run-domain-batch.py` with a one-line manifest, or the smoke runner):

    bash capstone/tests/fuzz/build-fuzz-program.sh cs7.c /tmp/cs7-O0.dom -O0

Next step, not yet done: a solo run with an instruction trace to see the last in-code pc,
then a per-function bisection -- build at -O2 with one function at a time marked
`__attribute__((optnone))` and run the whole set as one batch; the function whose -O0
codegen wedges is the reducer's starting point. Candidates by shape: a `cjalr zero, 0(ra)`
returning through a clobbered ra (a spill slot written through the wrong capability), or an
-O0 load/store of a capability through an address computed from a truncated value.
