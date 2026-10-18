# Result: the narrow-decline prototype's COMPILER arms, as pre-registered

Arms from `PREREGISTRATION-narrow-libcall.md` (pre-registered at `dc088366c8b5`, prototype at
`1f339723201e`). These are the compiler-only arms; the image arms are still unrun and the **void
gate** in the pre-registration binds them, not these.

Compiler: `clang 22.0.0git 01b62884c0c7`. Chosen by **feature probe, not path** — three toolchains
were on this host and only this one answers to the flag. The prototype's source is **identical**
between that built revision and the current branch head (both lines of `CapstoneNarrowLibcallC81`),
and `1f339723201e` is an ancestor of it, so the binary is the prototype rather than something
adjacent to it.

The flag is `-capstone-lower-narrow-libcall-c81` — `DEBUG_TYPE` is `capstone-lower`, so the
spelling comes from `--help-hidden`, not from the `cl::opt`'s suffix.

## Hazard 1, the go/no-go, is satisfied

The pre-registration says a build that does not guard `llvm.memcpy.inline` **is not to be run**,
because declining a forced-inline copy is a documented silent miscompile here. The built code is:

```c
if (CapstoneNarrowLibcallC81 && Limit != ~0U && Op.size() >= 16 && EffAlign < 16)
```

`Limit != ~0U` is exactly the discriminator the pre-registration names for the `AlwaysInline` case.
Guard present; the arms were run.

## The readings

Both arms are the **committed** `repro.c` and the **committed** classifier; the prototype runner is
a copy of `run.sh` differing by exactly one line, the added `-mllvm` flag (diffed, one hunk).

| function | opt | arm | baseline | prototype | changed |
|---|---|---|---|---|---|
| `copy32_unknown` | -O0/-O1/-O2 | plain | `SCALAR1` tags LOST | **`CALL` tags kept** | yes |
| `copy32_align8` | -O0/-O1/-O2 | plain | `SCALAR8` tags LOST | **`CALL` tags kept** | yes |
| `copy32_known` | -O0/-O1/-O2 | plain | `CAPS` tags kept | `CAPS` tags kept | **no** |
| `copy32_unknown_builtin` | all | both | `SCALAR1` tags LOST | **`CALL` tags kept** | yes |
| every already-`CALL` row | — | nobuiltin | `CALL` | `CALL` | no |

12 of 24 rows changed, every one of them a tag-stripping row becoming a call. **12 unchanged.**

**The pre-registered prediction holds**: `copy32_unknown` and `copy32_align8` become calls, and
`copy32_known` is unchanged `ldc`/`stc`.

**Hazard 3 holds, measured rather than asserted: `copy32_known` rows changed = 0**, byte-identical
instruction and capability counts (`8/10` at -O0, `8/6` at -O1/-O2) across both arms. The S-06
handshake's first block is undisturbed.

## One conversion BEYOND the prediction, recorded as an addition

`copy32_unknown_builtin` also becomes a call, in **both** the `plain` and `nobuiltin` arms. The
pre-registration predicted only the other two. This is the shape that stripped tags even under
`-fno-builtin` — an explicit `__builtin_memcpy`, which `-fno-builtin` cannot reach — so it was the
one case the existing workaround could not cover, and the prototype covers it. Desirable, but it
was not predicted, so it is written here as new information rather than folded into the prediction.

## Why the runner exits 1, and why that is not a failure

`run.sh`'s `EXPECT` table encodes **today's defective** behaviour, so each fixed row prints
`<-- UNEXPECTED, wanted SCALAR1`. The non-zero exit is the test asserting the defect is still
present. Read the direction, not the status: every "unexpected" row moved from *tags LOST* to *tags
kept*. Updating that table is a decision for whenever the flag's default changes, not part of this
result.

## The toolchain is STALE by the project's own gate — what that does and does not affect

Found after these arms were committed, while setting up the image arms:
`toolchain-fresh` reports `llvm-lane-c27` **STALE** — 206 ninja steps pending, 33 of them
compile/tablegen outside the revision chain — and `build-corpus-images.sh` will not produce an
artifact from it.

**What it does not affect: the readings above.** Both arms were compiled by the *same* binary and
differ only by the flag, so whatever state that binary is in cancels in the difference. The 33
out-of-chain steps the gate names are all `ObjCopy`/`DXContainer`, nothing in Capstone codegen, and
the prototype's own source was checked identical between the built revision and the branch head.

**What it does affect: any absolute claim.** "This is what the compiler does today" is not
supported by a stale build — only "this is what the flag changes in this build" is. The table above
is a difference, so it stands; nothing here should be quoted as dev's current behaviour.

**And it blocks the image arms**, which is how it was found: the corpus build gates on freshness
and refuses, correctly. A rebuild is required first — on this host that is `-j12`, not the `-j90`
the gate's message suggests, which is another machine's constant.

## What is NOT established here

Nothing about whether the direction is sound in a running image. The wholesale flag's fault (mruby
cause 24 inside `memcpy`, before case code) is what the narrow decline exists to avoid, and only
the mruby and SQLite arms can speak to that. Those remain **unrun**, and when they run the void
gate applies: an image where the prototype converts zero sites yields a result that is **VOID**,
neither a pass nor a failure, and the converted-site count must be recorded beside each verdict as
calls-gained over the unmodified build.
