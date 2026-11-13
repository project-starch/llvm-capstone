# CAPSTONE_MEMSET_CAP_ZERO on the compiler-lever arms' SDK: directed test

`ms-zero.c` runs the runtime's memset (called through a volatile pointer) over a buffer of tagged capabilities whose
words were poisoned first.
- Fills: 0, and 0x5a (the word path).
- Every head offset 0..15, and 26 lengths from 0 to 255: 832 cases.
- After a fence and again after a 1 MiB eviction sweep, it requires:
  - the fill inside the range;
  - every granule the range does not touch byte-identical to before, still tagged, and still naming its cell;
  - every granule the range touches untagged.
- Controls: the same readout with no memset finds every tag set, and `-DMS_MUTANT` (a fill one byte short) must fail.

**Not checked:** bytes outside the range in a granule the range touches. Once the fill clears that granule's tag, its
other bytes read as the twin's raw capability encoding, not the cursor a plain load of the tagged capability returned.
The switch-off memset shows the same, in the same 686 cases, so this is QEMU's memory model, not memset.

**How it runs.** The native route, as the SPEC images do:
- the program is built with the arm SDK's capstone-cc and staged in a full copy of a native SPEC stage, replacing its
  binary;
- `perf/faultesc/run.sh` runs that stage on the native twin;
- stdout comes back as `results/inp.out`.
(The virtual kit's capstone-vexec cannot run a native-SDK image: its first capability access faults with cause 24.)

## Results (2026-11-14; llvm-c8 with FRM's flags; natio images)

| arm | memset | ms-zero |
|---|---|---|
| switch off (natevict SDK) | no stc | 4/4 checks, 832 cases, 0 misses |
| switch on (perf/memset SDK) | 5 stc | 4/4 checks, 832 cases, 0 misses |
| switch on, `-DMS_MUTANT` | 5 stc | FAIL, 656 misses (bytes inside the range) |

hosted-contract on the native route reads 37 pass and 8 fail on both arms, with identical check lines. The 8 are file
I/O checks that expect the virtual stage's paths. The SDKs' runtime archives are identical by disassembly except
`string_bounds_safe.c.obj`, and their libc archives are identical.
