# Recent commit audit: June benchmark and capability work

Audit date: 2026-06-25.

Range audited: `161d32974191` through `e6f35b2b95e9` on
`capstone-bootstrap`. This covers the recent backend fixes, BEEBS completion,
RV8 C-suite bring-up, aggregate runners, capability-global initialization, and
CoreMark restoration work.

## Executive summary

No release-blocking correctness issue was found in the audited commits. The
large decisions are technically coherent and match the current verified baseline:
Capstone CodeGen lit passed 29/29, RV8 passed 7/7, BEEBS passed 82/82 in the
aggregate after retrying transient QEMU infra flakes, and CoreMark again prints
"Correct operation validated" after the multi-module cap-init fix.

The most important architectural decisions are:

- `9bb828d27b9c`: fix `va_list` by treating the cursor as a capability and
  advancing by 16-byte slots.
- `a3126729e88d` and follow-ons: use a small explicit soft-float/libm slice for
  domain benchmarks instead of importing a full libc.
- `30601e06829d` and `e7d5b87656e1`: materialize capability globals through
  compiler-emitted runtime stores, with a PC-relative multi-module init table.
- `5a00b4c78b11`: fix stack-passed capability arguments, a concrete ABI tag-loss
  bug.
- `97fe8317b5a8`: add isolated parallel BEEBS aggregate mode with quiet logs and
  structured infra-flake retry.

Issues found during this audit are low or documentation-level:

- `CapstoneCapGlobalInit.cpp` still described the new registration as
  `llvm.global_ctors` / `.init_array`; the code actually uses
  `.capstone_cap_init`.
- `current-next-step.md` still carried the pre-multi-module weak-default wording
  before the update paragraph.
- `capability-globals-init-decision.md` still said the capability-global fix
  alone did not make `dtoa` pass; that became stale after `216396dbafd4`.
- Security review flagged the writable QEMU 9p host share as a medium risk if
  the harness is used for untrusted domains, but the benchmark certification
  workload is trusted and uses `/tmp/capstone`.

## Backend and capability lowering

### `161d32974191` - sub-capability aggregate memcpy

Decision and implementation: fix aggregate copy behavior for object sizes that
do not naturally fill a capability slot. The lit coverage in
`aggregate-memcpy-align.ll` makes the intended lowering visible.

Assessment: sound and important. On a tagged architecture, "small memcpy" is not
just an optimization detail: scalar byte movement and capability-slot movement
have different tag behavior. The commit keeps the compiler from corrupting
neighbor state when only part of a capability-aligned aggregate is copied.

Risk: still leaves the broader policy question of when byte-wise copies should
preserve or intentionally strip tags. That belongs in the capability authority
audit track.

### `9bb828d27b9c` - `va_list` capability lowering

Decision and implementation: make `VAARG` and `VACOPY` custom lowering, make
`VASTART` store an address-space-200 capability cursor, and advance each vararg
slot by the 16-byte capability slot size. CoreMark's assembly trampoline was
removed after the backend fix validated.

Assessment: strong fix. It addresses two independent backend bugs: tag loss from
scalar `sd`/`ld` of the variadic cursor, and wrong 8-byte cursor stride for
stack-saved varargs. The new `vararg.ll` regression tests the actual Clang IR
shape.

Risk: GISel remains conceptually parallel and should not be assumed fixed. This
is worth mentioning in future compiler-status docs.

Research value: high. This is a compact example of how a correct IR-level
capability type can be broken by target lowering that asks for the wrong pointer
type.

### `52033647573d` - i128 non-vector shift fallback

Decision and implementation: add a general constant-shift fallback in
`lowerScalarI128Shift` for patterns such as pointer-difference division lowering.

Assessment: sound. It unblocks real benchmark code without weakening the
capability-forging guard. The lit extension in `i128-xlen-lowering.ll` is the
right level of regression coverage.

Risk: the fallback should remain narrow to scalar constant shifts; future
changes must not turn arbitrary i128 integer operations into capability
constructors.

### `da5ef4c9c018`, `5a00b4c78b11` - stack-passed capability args

Decision and implementation: diagnose RV8 `norx` as an untagged 9th+ capability
argument path, then fix outgoing stack-slot address derivation to preserve
capability tags.

Assessment: strong. The diagnosis commit is valuable, not noise: it documents
the observed ABI failure class and includes a reduced runtime repro. The fix is
well scoped and has a CodeGen regression.

Risk: other ABI paths that move capability-shaped values through scalar slots
still deserve audit coverage, especially returns, by-value aggregates, inline
asm, PHI/select, and memcpy-like lowering.

Research value: high. This is another concrete "integer/capability confusion"
case study for the paper.

## Capability globals and startup architecture

### `349451b77c60`, `0a602ede55a0`, `30601e06829d`

Decision and implementation: validate that runtime C assignment produces tagged
capability globals, extend `.gct` metadata to arrays, then add the
`CapstoneCapGlobalInit` ModulePass that emits stores into capability global
slots before `domain_main`.

Assessment: the constructor-codegen decision is pragmatic and technically
better for current domains than a hand-written `.gct` runtime consumer. It uses
normal compiler lowering to derive and store capabilities, so the runtime does
not need to understand every global shape. Leaving the static initializer intact
and making the stores volatile is the right choice because `.gct` metadata still
needs the original initializer and the untagged bytes are overwritten before
use.

Risk: the first implementation used a strong external `__capstone_cap_init`,
which was acceptable for single-module domains but regressed CoreMark. This was
fixed later in `e7d5b87656e1`.

Research value: high. This is one of the cleanest paper-worthy design decisions
in the series: tags cannot live in ELF, so the compiler synthesizes code that
materializes them from valid authority at runtime.

### `216396dbafd4`, `18119046df29`

Decision and implementation: resolve the last BEEBS deferred cases: `dtoa` by
combining capability-global materialization with a 16-byte-aligned arena, and
`trio-snprintf` by reusing the now-working `va_list` and capability-global
infrastructure.

Assessment: sound, with an important distinction: `dtoa` did not pass because
one magic workaround hid all problems; it needed two separate fixes with
different causes. The docs should preserve that split.

Risk: `dtoa` uses deep floating/string conversion code and local allocator
assumptions. It is validated as a benchmark, not as proof of full libc support.

### `e7d5b87656e1`, `e6f35b2b95e9`

Decision and implementation: make per-module cap-global init functions internal
and emit a `.capstone_cap_init` table of PC-relative offsets. `start.S` derives
each initializer's runtime code capability as `gp + (entry_runtime + offset)`.
Weak GCT begin/end markers avoid multi-module duplicate symbols.

Assessment: correct for this loader model. Absolute `.init_array` entries are
not viable because the domain is loaded at a runtime base and does not process
load-time relocations. Internal functions avoid symbol collision without adding
runtime name lookup. The linker-bounded table and startup loop are simple and
auditable.

Risks:

- The table is trusted compiler/linker output. Adversarial objects can add
  entries and trigger early calls. This does not add new authority when linked
  objects are already trusted, but it is not a sandbox boundary.
- Weak GCT markers are safe for current runtime behavior, but future tooling
  should not interpret a selected weak pair as bounding all `.gct` records.
- Comments had drifted and referred to `llvm.global_ctors` / `.init_array`.

Research value: high. The PC-relative table is an important consequence of the
domain loader's no-relocation model.

## Floating point, libc boundary, and BEEBS expansion

### `8a4559693112`, `a3126729e88d`, `c06b59d36f9b`

Decision and implementation: add `compress`, introduce the soft-float/libm path
with `cubic`, then factor reusable soft-float builtins for `sqrt`.

Assessment: sound. `compress` showed that some previously deferred entries had
become ordinary bring-up work. `cubic` correctly forced the project to confront
runtime libcall lowering and minimal libm rather than faking the result.

Risk: soft-float/libm routines must remain explicitly scoped. This is not a
general hosted libc.

### `1daba2f7a007`, `0c259bca869d`, `89b6d34c4ebe`

Decision and implementation: expand through linear algebra and FP benchmarks
with adapted oracles, correctly-rounded `sqrt`, and shared libm naming.

Assessment: good engineering. The exact-oracle pattern is defensible, and the
rename from benchmark-specific `beebs_cubic_libm.c` to shared
`beebs_softfloat_libm.c` correctly reflects reuse.

Risk: source adaptations that move local const arrays to static const avoid a
known backend issue rather than fixing it. The docs correctly keep that bug in
the compiler backlog.

### `ac709d4892b9`, `97fe8317b5a8`

Decision and implementation: tighten BEEBS oracles and make aggregate runs
quiet by default, then add isolated parallel aggregate mode.

Assessment: strong workflow decision. Quiet logs solve the agent-token problem,
while isolated per-benchmark work directories and `BEEBS_FETCH_READONLY=1`
address the race hazards that made naive parallelism unsafe.

Risks:

- Parallel QEMU still exposes host-resource flakes; certification should report
  infra retries explicitly.
- The runtime harness writable 9p share is acceptable for trusted benchmark
  code, but should be guarded if untrusted domains are ever tested.

### `0381175a4133`, `de580f57aa3b`, `3ce8537a8b67`, `4ca17bf4f61d`,
`006c2b43e703`, `993c6d07470e`

Decision and implementation: complete the soft-float/libm-only class, add
newlib math benchmarks, `stb_perlin`, `matmult-float`, `whetstone`, `fasta`,
and the simple `janne_complex` case. Record the AOR-vs-picolibc decision.

Assessment: the sequence shows disciplined scope control. The team did not pull
in a broad libc to satisfy a few functions. Instead, it added the computation
slice required by benchmark code and left OS-like services out of scope.

Risk: the freestanding string/memory routines are appropriate for computation,
but capability-containing object copies through byte buffers remain a future
semantic question.

Research value: medium to high. The exact-oracle method and "small explicit
runtime instead of full libc" decision are useful in the evaluation section.

## RV8 suite

### `5f8692b75bd8`, `97131374070d`, `f06bfc76b203`, `012b508caeaa`,
`67b385e17323`, `c02ec1be7b2a`

Decision and implementation: stand up RV8 as a split host/domain suite and add
seven C benchmarks with reduced workloads, stubs, a bump allocator, and
self-contained oracles.

Assessment: good suite construction. The RV8 work reuses BEEBS infrastructure
where appropriate and keeps domain adaptations explicit.

Risk: workload reductions must stay documented so RV8 results are not confused
with performance claims for the original suite.

### `0b7bfb23f6dc`, `3f286976e454`, `dcc9c0cee120`

Decision and implementation: scaffold `norx`, expose and later fix a real
backend tag-loss bug, defer C++ `bigint`, and add `run-all-rv8.sh`.

Assessment: correct triage. `norx` was not papered over; it forced a backend
fix. `bigint` is correctly deferred because it needs C++ runtime/STL and has a
separate `new` expression backend crash.

Risk: `run-all-rv8.sh` retries any first failure once, not only structured infra
flakes. This is acceptable for certification convenience, but reports should
mention whether a retry occurred.

## Aggregate gates and validation docs

### `32c4c6419723`, `a166e0d6ce19`

Decision and implementation: add serial aggregate wrappers for HostCall,
`null_blk`, and BEEBS, then record full BEEBS aggregate validation.

Assessment: sound. Aggregate gates make the current workflow reproducible while
leaving individual wrappers as authoritative diagnostic entry points.

Risk: none significant. The scripts should continue to stop on first hard
failure and print the active child script/log path.

## Security review notes

Security review of the recent cap-init and harness surface found no high issue.
The main findings are:

- Medium if used on untrusted code: `run-domain-smoke.py` exports caller-chosen
  `--share-dir` as a writable 9p mount with `security_model=none`. Current
  wrappers use trusted benchmark code and `/tmp/capstone`, but the harness should
  reject dangerous share paths or require an override before it is used on
  adversarial domains.
- Low: positional domain filenames are interpolated into a guest shell command.
  Current filenames are fixed and trusted; strict filename validation or
  `shlex.quote` would harden the generic harness.
- Low: `.capstone_cap_init` accepts all linked section entries. This is fine for
  trusted compiler output and should be documented as such.
- Low: weak GCT markers are ambiguous for whole-program inspection if a future
  consumer revives `.gct`.

## Recommended follow-ups

1. Keep `design/research-decisions-log.md` current after future non-obvious
   backend/runtime decisions.
2. Commit or explicitly discard the untracked
   `design/capability-provenance-threat-model.md`; it is valuable, but its
   current untracked state makes the handoff unclear.
3. Add a small hardening patch for `run-domain-smoke.py` before using it with
   untrusted domains: constrain `--share-dir` to `$CAPSTONE_TMP_ROOT` by default,
   quote guest shell paths, and document the override.
4. Continue the capability authority audit with tests for laundering through
   scalar memory, PHI/select, by-value aggregates, inline asm, and function
   boundaries.
