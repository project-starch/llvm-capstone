# C-81 (provisional number) — a constant-size `memcpy` of unprovable alignment strips capability tags

**Sibling issues, so a reader with the wrong symptom is redirected immediately.** Three defects
share the symptom "a copy lost a tag" and have three different mechanisms:

* **C-78** — a copy that IS provably 16-aligned, whose size is not a multiple of 16. Fixed
  2026-10-13. Its fix branch is gated on `Op.isAligned(Align(16))`, so it cannot reach this case.
  If your operands are provably 16-aligned, you want C-78.
* **C-69** — a capability-**TYPED** load/store carrying `align 1`, legalized to 16 `lbu` into a
  stack temporary and then an `ldc`. A *different lowering stage*: unaligned-access legalization of
  a `c128` access, not `findOptimalMemOpLowering`. **C-69's proposed fixes cannot reach this case**,
  and that asymmetry is the substantive point rather than bookkeeping: C-69 is fixable precisely
  *because the type says "capability"* ("refuse to lower it", "treat capability-typed accesses as
  naturally aligned"), whereas here the IR is type-erased and carries no capability type to key on.
  C-69's own README lists "a memcpy-shaped idiom" among the ways to produce an under-aligned
  capability access, which is about producing an align-1 capability-TYPED access — a matching shape
  is not a matching mechanism, and this is not an instance of it.
* **this entry** — a type-erased `llvm.memcpy` whose operands carry `align 1`.

The number above is provisional — the registry owner assigns it.

## What happens

A constant-size `memcpy` over a region that holds a capability is expanded into scalar chunks
sized to the copy's *provable* alignment. When the compiler cannot prove 16-alignment — the normal
case for `void *` parameters — the chunks are one byte, and a plain `sb` never carries a tag. The
copy succeeds, the bytes arrive, and every capability in the range arrives untagged. Dereferencing
it then faults with cause 24, or worse, reads as a plausible non-capability value.

## Measured

`run.sh`, compiler `85b1daddf230`, every arm matching its recorded expectation (exit 0):

| function | `-O0`/`-O1`/`-O2` plain | with `-fno-builtin` |
|---|---|---|
| `copy32_unknown` (`void *` params) | 32x (`lbu`+`sb`) — **tags LOST** | calls `memcpy` — kept |
| `copy32_known` (`struct cap16 *`) | `ldc`/`stc` — kept | calls `memcpy` — kept |
| `copy32_unknown_builtin` (`__builtin_memcpy`) | 32x (`lbu`+`sb`) — **tags LOST** | 32x (`lbu`+`sb`) — **tags LOST** |

Identical at `-O0`, `-O1`, `-O2`, `-O3`, `-Os`, `-Oz` and `-Og`.

**MEASURED on both revisions, not inferred — and the second compiler is easy to miss.** It is not
under `/tmp/capstone/llvm-*`; a glob of that directory concludes it is absent, which both I and the
auditor did at different times. It is here:

    /tmp/capstone/virt-stack/llvm-build/bin/clang   -> bd372e25f5fb
    /tmp/capstone/llvm-dev-current-20261017/bin/clang -> 85b1daddf230

printed by `clang --version` in the same command as the numbers. The two emit **byte-identical**
`copy_pair`/`copy32_*` bodies in every arm (body sha `0f832598dde9` plain, `9871b6a90cc3` under
`-fno-builtin`). Corroborated by source: the executed branch is character-identical between the two
revisions, the only differences in that function being C-78's branch being added and one
`unsigned` -> `uint64_t` widening elsewhere in it.

**So this is long-standing, not a regression.** A claim that it WAS a regression between those two
revisions was made and withdrawn on 2026-10-17; the lane branch carries that retraction. Do not
cite `scratchpad/shape-old_bd372e25.s` / `shape-kit_85b1dadd.s` as evidence for anything: all four
of those files contain the `-fno-builtin` body, so a reader opening them sees zero `lbu` and
concludes the opposite.

The run-time half is `runtime/virtual/hosted-contract.c`'s
`memcpy_constant_size_unknown_alignment`, which reads the copied capability's tag as 0 without
`-fno-builtin` and 1 with it. **Careful with that check**: `copy_pair` there is `static` with a
single call site over file-scope statics, so at `-O1` and above interprocedural constant
propagation substitutes the operands in and the compiler *can* then prove 16-alignment — the check
passes and stops testing what its name says. `noinline` does not prevent that. This reproducer
takes its operands as external-linkage parameters precisely so no pass can specialise it.

## Where it is

`CapstoneTargetLowering::findOptimalMemOpLowering`, in
`llvm/lib/Target/Capstone/CapstoneISelLowering.cpp`. **Cited by guard, not by line number: that
file's own note forbids line citations in it** ("Do not cite line numbers in this file at all: they
rot on the next edit, and this comment has now got them wrong twice in opposite directions. Find
the three `Op.isMemcpy()` guards in `findOptimalMemOpLowering` by name"), and warns the three are
"THREE DIFFERENT MECHANISMS, not one avoidance wearing three hats".

The branch is the **last** of the three, the one guarded by

    Op.isMemcpy() && Op.size() != 0 && !(Op.isAligned(Align(16)) && (Op.size() % 16) == 0)

and which falls through to `TargetLowering::findOptimalMemOpLowering` when its chunk budget is
exhausted. It caps the unit at 8 bytes (`// never i128 on this path`), sets `EffAlign` from the
source alignment and then narrows it by the destination's **only if `Op.isFixedDstAlign()`** — for a
fresh-alloca destination it is the source alignment alone — then divides the unit down to it.

The other two guards both require `Op.isAligned(Align(16))` and return early, so an
unprovable-alignment copy reaches this branch by construction. They are distinct and must not be
conflated: the **second** assigns `c128` chunks to a 16-aligned 16-multiple copy and predates C-78;
the **third**, C-78's own, handles a 16-aligned copy whose size is not a multiple of 16.

**Measured premise.** The IR for `copy32_unknown` carries `align 1` on both operands at every level
(`llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 1, ptr addrspace(200) align 1, i64 32, ...)`),
and the control carries `align 16` on both. So `EffAlign` is 1 and 32 one-byte chunks follow.

**It is provably this branch and not the generic fallback.** `MaxStoresPerMemcpy` is 8
(`CapstoneProcessors.td`), and the generic implementation refuses once `NumMemOps` exceeds `Limit`,
so the generic path cannot have emitted 32 inline memops — it would have returned false and
produced a libcall. The branch's own chunk budget gives a signature nothing else in the compiler
has: at 64 bytes the copy inlines as 64 memops, and at 65 it becomes a `memcpy` call. Both
observed, at `-O0` and `-O2`.

**Why it must not simply use capability units.** An `align 1` operand cannot be given a 16-byte
`ldc`/`stc` without a run-time alignment check — which is exactly why routing to musl's `memcpy`
preserves the tag. **Do not repeat the file's older rationale** about a misaligned-source `i128`
load being mis-legalized to an `i64` load zero-extended against a 16-byte `stc`: that note's own
follow-up records it as closed, because `i128` was retired as the capability carrier in favour of
`c128` and is no longer a legal type (`addRegisterClass(MVT::c128, &Capstone::GPCRRegClass)` is the
only 128-bit register class). It was true when written and describes a tree that no longer exists.
Whether a misaligned-source **`c128`** unit would misbehave as the old `i128` one did is
UNRESOLVED — it cannot be tested without changing the code.

## What a fix has to do

**The criterion is size >= 16 with operands not provably 16-aligned, whatever the unit.** Not
"byte-granular": a region the compiler knows only as 8-aligned can be 16-aligned at run time and
hold a capability, so an `EffAlign` 8 `ld`/`sd` expansion strips tags exactly as `sb` does. The
real images bear this out — the faulting-only runs are 55 at unit 1, 18 at unit 8 and 9 at unit 4,
so a fix written for unit 1 alone would miss a third of them. Writing it narrowly would repeat
C-78's mistake of fixing the reproducer's shape instead of the defect.

Two candidate directions:

1. **A run-time alignment test**, with the capability path behind it. Available now, and the
   default recommendation.
2. **Emit the libcall.** This is CHERI's rule and is why `-fno-builtin` works at all — musl's
   `memcpy` checks alignment at run time and copies a 16-aligned granule as a capability.

**Direction 2 is currently BLOCKED by a measured failure, not by a hypothesis.** The in-tree flag
that implements it wholesale, `-mllvm -capstone-lower-memops-via-libcall` (`CapstoneMemOpsViaLibcall`),
carries this status immediately above its declaration:

    STATUS: DOES NOT WORK YET -- do not enable it expecting a working domain. With the flag on,
    SQLite faults under QEMU at `helper_cscincoffset: Assertion rs1_v->tag failed`, i.e. a
    cincoffset on an UNTAGGED register, immediately after domain entry.

Corroborated in `docs/ref/ISSUES-ARCHIVE.md`. **So do not adopt that flag as a workaround**, even
though at compile time it is strictly more complete than `-fno-builtin` (it converts
`__builtin_memcpy` too — see the table below). "The copies become calls" says nothing about whether
the resulting domain runs, and its own status says it does not.

Two things keep direction 2 alive as a question rather than closing it:

* **A contradicting measured PASS.** `docs/plans/compiler-validation-plan.md` records
  "W-05 memops-via-libcall (RV8 -O2): 7/7 AGREE, images differing". So the status is "fails SQLite,
  passes RV8", neither retracted.
* **The SQLite record is confounded and may be stale.** It is from 2026-08-11 and the flag then
  "had to be paired with `BEEBS_LDC_HIGH_HALF_FIXUP` in the library", an arm the same comment says
  **no longer exists** (S-06 was fixed in silicon 2026-08-15 and the workaround reverted). It also
  predates the C-79/C-80-era fixes.

**The deciding experiment** is therefore the W-05 matched pair in
`docs/plans/compiler-validation-plan.md`, run on today's tree with the v6 flags, asserting the two
`.dom` sha256s differ. If it reads clean, direction 2 is available; if the fault is live, direction
1 is the deliverable.

**The `gp` concern is NOT the gate, contrary to an earlier version of this folder.** The note that a
compiler-generated libcall "resolves its target through `gp`" is labelled "HYPOTHESIS, NOT
ESTABLISHED ... Confirm before acting on it". And it does not apply to this profile: under the v6
flags (`-capstone-gp-free -capstone-image-gp`) the emitted libcall is
`auipc` / `addi` / `mv t1` / `jr t1` with **no `gp`**; the `gp`-resolved shape
(`cincoffset a3, gp, a2`; `cjalr`) belongs to the non-gp-free ABI. Compiler-generated `mem*`
libcalls also already occur on the default path for copies past the chunk budget, so they are not
structurally fatal.

## The libcall flag has TWO uses; one is valid and one is withdrawn

Keep these apart, because the flag appears in this folder for both reasons:

* **As an enumeration INSTRUMENT: valid.** Compile with
  `-mllvm -capstone-lower-memops-via-libcall`, never run the result, and diff it against the plain
  build. Every copy that becomes a call was inline-expanded before, including `__builtin_memcpy` and
  compiler-synthesised copies that `-fno-builtin` cannot reach. The positive control is
  `copy32_unknown_builtin` becoming a call. The flag's withdrawn status concerns *running a domain
  under it*, which this use never does.
* **As a WORKAROUND: withdrawn**, pending the W-05 matched pair. See above.

## `-fno-builtin` is an INCOMPLETE workaround — do not record it as the resolution

The third row of the table is the point: `__builtin_memcpy` byte-expands **with** the flag. The flag
only keeps *explicit* `memcpy` calls away from the inline expansion; it does not touch
`__builtin_memcpy`, and by the same path it does not touch copies the compiler *synthesises* —
struct assignment, by-value aggregate arguments, local array initialization. ISSUES.md says the same
of C-78 at :8574-8577.

It also has a cost beyond this defect, and the registry already records both halves of it. The SDK
**used** to pass `-fno-builtin`; dropping it is what exposed C-78 (":8556", found "when the SDK
stopped passing -fno-builtin") and C-79 (:8498), whose own "why it was not seen before" is exactly
that the flag "disables these" libcall simplifications (:8511). So:

* re-adding the flag is a **reversion to the SDK's older behaviour**, not a new configuration;
* it re-masks *this* defect, which is unfixed, and that is the point of arm 6;
* C-78 and C-79 are both FIXED (2026-10-13), so it no longer masks a live bug — it discards
  simplifications that now work. That is a codegen-quality cost, not a correctness one.

Any gate that adopts it should say which simplifications it gives up.

So a corpus reading of "all cells restored under `-fno-builtin`" means *those* copies were explicit
`memcpy` calls. It is evidence the workaround covers the measured set, not that the images are safe.

## The scanner, and the real images: the known instance IS this defect

`scan-memcpy-expansion.py` finds straight-line scalar load/store runs moving >= 16 bytes in
sub-capability units. Its positive control (`--self-test`, this folder's `repro.c`) fires in both
directions and at both units.

**THE SCANNER IS NOT AN ENUMERATION, AND ITS COUNTS HERE ARE LOWER BOUNDS.** Counting instead the
`mem*` call references a libcall twin GAINS, per function, the board lane measured (dev
`935196067e79`, `capstone/tests/rtl-smoke/c81-silicon-exposure-2026-10-17/`):

| image | expansions the twin diverts | this scanner's plain-only runs |
|---|---:|---:|
| SQLite 02 | 48 | 2 |
| SQLite 05 | 47 | 2 |
| set 1 controls | 12 | 8 |
| set 1 heap-00 | 13 | 9 |
| mruby | 2,957 | 89 |

Most of the remainder are `ldc`/`stc`-width or under 16 bytes, which this tool is right to ignore —
but the gap is far too large for the diff to be read as a count of anything. **To enumerate, count
the calls the twin gains** (`memcalls2.py`), then judge each candidate with `granule-check.py`,
which flags a run whose stores cover 16 contiguous bytes from one base and has its own positive
control. `rescan-nosplit.py` is the equivalent of the fix described below.

**A defect this tool had, with its measured cost, because the lesson generalises.** `BREAKERS` used
to include `c[a-z]+`, which matched `cincoffset`/`cincoffsetimm` — the pointer arithmetic a copy does
*between its own chunks* — so one copy was split into sub-threshold pieces and dropped. On two
subobject images that hid **30 runs each** (91 -> 121, 82 -> 112). **The positive control could never
have caught it**: `repro.c`'s copies have no intervening capability arithmetic, so the check was
keyed to one shape and silently missed another. Fixed, and `--self-test` now carries a synthetic
two-sided regression test — 32 one-byte stores with a `cincoffsetimm` in the middle must report as
ONE run, and a `beq` in the middle must still split one.

Re-measuring the nine-case diff with the fix changes the counts and not the conclusion: unit-1 runs
55 -> 56, unit-4 9 -> 20, unit-8 18 -> 18. Per case the byte-granular count stays at 6 (case 01
7 -> 8). **So the table below is a lower bound on a real asymmetry, not a census.**

**A second known limit:** `--diff` keys runs by SHAPE, not by function, because these images are
stripped and the two builds renumber `.Lpcrel_hiNN`. Two distinct copies with the same shape are
therefore indistinguishable, and can cancel in the set difference.

**Use it as a DIFF between two builds of one source, not as a scan of one image.** A single
image's hit count is dominated by ordinary struct and array code at unit 8; the informative
quantity is which runs one build has and the other does not.

Nine FFmpeg subobject cases, each faulting build against its arm-6 (`-fno-builtin`) twin, cited by
image hash:

| case | faulting | clean | runs f | runs c | only-f | only-c | faulting-only by unit |
|---|---|---|---|---|---|---|---|
| 00 cbs_h265_pic_timing | `bdf538f4` | `ffd2170b` | 91 | 82 | 9 | 0 | 6x unit1, 1x unit4, 2x unit8 |
| 01 vulkan_hevc_refpicset | `987ececc` | `69e01ee5` | 93 | 84 | 10 | 1 | 7x unit1, 1x unit4, 2x unit8 |
| 02 vulkan_hevc_dpb | `699502ad` | `94efd600` | 92 | 83 | 9 | 0 | 6x unit1, 1x unit4, 2x unit8 |
| 03 aac_arith_last | `24f998bb` | `05bb4aaa` | 91 | 82 | 9 | 0 | 6x unit1, 1x unit4, 2x unit8 |
| 04 rtsp_control_url | `3b3c04bc` | `46793a07` | 91 | 82 | 9 | 0 | 6x unit1, 1x unit4, 2x unit8 |
| 05 vulkan_av1_tile_sizes | `f028d676` | `16e5bfb6` | 91 | 82 | 9 | 0 | 6x unit1, 1x unit4, 2x unit8 |
| 06 vvc_entry_point_start_ctu | `fa542b53` | `63651fbc` | 91 | 82 | 9 | 0 | 6x unit1, 1x unit4, 2x unit8 |
| 07 hlsenc_key_uri_strlcpy | `0b37fef0` | `3deb5adb` | 92 | 83 | 9 | 0 | 6x unit1, 1x unit4, 2x unit8 |
| 08 cbs_h266_col_width_val | `cf8edbfb` | `2c6f712b` | 91 | 82 | 9 | 0 | 6x unit1, 1x unit4, 2x unit8 |

55 unit-1 runs across the nine, every one moving exactly **16 bytes — one capability, expanded one
byte at a time.** Each faulting build is a near-strict superset of its clean twin (only case 01 has
a single clean-only run), which is what a flag that turns expansions into calls must produce.

So **the FFmpeg subobject fault's mechanism is this defect**, established from the artifacts without
spending a board run. What the diff shows precisely is that `-fno-builtin` removed 6-7
byte-granular 16-byte expansions per case; it does not single out which one stripped the tag read
at `refstruct_pool_uninit+0x78`. The attribution rests on 16 bytes being exactly one capability,
byte-expanded, present only in the builds that fault, consistently across nine independent cases.

`controls.dom` is **byte-identical** between the arms (`42df2b60`), so its 0/0 reading is trivially
expected and is NOT a negative control — it is excluded from the table rather than cited.

**A RETRACTION, recorded because the mistake is instructive.** An earlier version of this section
said the scanner could not see the known instance, on the strength of zero unit-1 runs in
`corpus-sil/share-silicon-pool-stock-r3/ffmpeg-pool-0*.dom`. That was void twice over: those images
do **not** fault (silicon reads MISSED 4 with no fault), and they were built `-ffreestanding
-fno-builtin`, so an absence of explicit-memcpy expansions in them is a tautology. I had selected
them by NAME, never checking the verdict record or the build flags. The project's own rules both
cover it — cite a board result by its image hash rather than its label, and a negative result is
void unless the input held something the check could find.

## Scope

`-fno-builtin` was introduced on 2026-10-17 as `vp2-prep/sdk-arm-6`. Of the 19 `sdk.json` files on
the build host, **1 carries it and 18 do not** — including every Tier 3 virtual SDK (`v6-fix`,
`v6-pilot`, `v6-561`, `v6-b61`, `v6-instr`, `dev-proof`). So every image built before that arm was
built without it.

**The corpus runner's record has the same gap, and this one is worse because two different
builds carry IDENTICAL provenance.** For the nine-case diff above, the plain bundle's
`virtual-malloc/inputs.json` and the arm-6 twin's are byte-for-byte identical in everything that
describes the build:

* `cc_args` — identical, and **neither contains `-fno-builtin`**;
* `build.flags` — `["-O0", "-DCORPUS_VIRTUAL"]` on both;
* `build.sdk_identity` — `["virtual-mallocng", "virtual"]` on both, i.e. heap and profile only,
  which does not name the SDK;
* `platform.compiler`, `.launcher`, `.module`, `.qemu`, `.rootfs` — all identical.

Yet the images differ (`bdf538f4` vs `ffd2170b`). The flag came from the arm-6 **SDK's** `sdk.json`
codegen, and nothing in the record captures that. So `inputs.json` cannot answer "was this the arm-6
build", and a re-run diffed against the record would read as unchanged. Recording the SDK's codegen
list, or a hash over it, in both `inputs.json` and `manifest.json` would close both gaps.

**RETRACTED: arm 6 IS a one-variable pair, so the nine-case diff needs no such caveat.** An
earlier version of this paragraph said the arm-6 SDK moves the runtime archive too, so the
byte-granular runs might lie in the runtime rather than the case code. That was wrong, and wrong
because I compared against the WRONG baseline — `sdk-abi-v6-20261017`, which did not build the
Tier 3 kit. Against the kit's actual baseline, `vp2-prep/v6-fix/out/sdk`, verified file by file:

| SDK | `libapplication-runtime.a` | `libcapstone-application-builtins.a` | `spawn-shell.o` | `capstone-cc` |
|---|---|---|---|---|
| `v6-fix/out/sdk` (baseline) | `855f843c750a9838` | `4dff5d9426bdcffe` | `4c7cc33f439e1c11` | `7f3077ae894a9edb` |
| `sdk-arm-6` | `855f843c750a9838` | `4dff5d9426bdcffe` | `4c7cc33f439e1c11` | `7f3077ae894a9edb` |
| `sdk-abi-v6-20261017` (NOT the kit's) | `2955b33c0253845f` | `4dff5d9426bdcffe` | absent | `7f3077ae894a9edb` |

`diff -rq` between the baseline and arm 6 reports only `ARM.txt` and `sdk.json`, and `sdk.json`'s
only codegen difference is `-fno-builtin`. **Every prebuilt object is byte-identical**, so the
nine-run difference cannot be in the prebuilt runtime; it is in what `capstone-cc` compiled under
the differing codegen, i.e. the application side. The diff attributes correctly as it stands.

**There is no retrospective artifact test for this, and that is a recording gap.**
`ports/common/application/build.py`'s `manifest.json` (:172-188) records `compiler_sha256`,
`runtime_archive_sha256` and the revisions, but **not the codegen flags**. And the archive hash
cannot stand in for them: `sdk-arm-1` (no `-fno-builtin`) and `sdk-arm-6` (with it) produce a
byte-identical `libapplication-runtime.a` (`855f843c750a9838`). Checked two-sidedly — the arms must
differ for the hash to discriminate, and they do not. Recording the codegen list in the manifest
would close this.
