# Research decisions log

Durable list of implementation decisions that are interesting enough to cite in
the paper or revisit during design write-up. Each entry cites the commits that
introduced or materially changed the decision.

## Capability provenance and tag preservation

### Capability globals: constructor-codegen, then a PC-relative init table

Commits: `349451b77c60`, `0a602ede55a0`, `30601e06829d`,
`e7d5b87656e1`, `e6f35b2b95e9`.

Decision: materialize initialized capability globals by generating ordinary
runtime stores, not by teaching the loader to consume `.gct` metadata. A later
multi-module fix made each initializer internal and registered it through a
PC-relative `.capstone_cap_init` table.

Why it matters: capability tags cannot be represented in the static ELF image.
The constructor-codegen path turns that fact into ordinary compiler output:
derive a tagged capability from the domain root, `delin` as needed, and `stc`
the original global slot in place. The PC-relative table is important because
the domain does not process load-time relocations; absolute `.init_array`
entries are stale after runtime loading.

Limits: this preserves provenance from `gp`, but current capabilities inherit
broad root bounds. It solves tag materialization, not object-granularity bounds.
The `.capstone_cap_init` table is trusted object input, not a validation
boundary against malicious linked objects.

### `va_list` lowering as a tag-loss case study

Commit: `9bb828d27b9c`.

Decision: lower `VASTART`, `VAARG`, and `VACOPY` with the address-space-200
capability pointer type and 16-byte capability slots instead of generic scalar
pointer expansion.

Why it matters: the previous lowering stored and reloaded the variadic cursor
with scalar `sd`/`ld`, dropping the tag, and advanced by 8 bytes for `long`
instead of the 16-byte capability slot size used by the argument save area. This
is a concrete example of integer/capability confusion in the backend: the IR was
correct, but the SelectionDAG type choice made a capability travel through a
scalar path.

Limits: the fix covers the SelectionDAG path used by current benchmarks. GISel
has the same conceptual risk and remains a separate follow-up.

### Stack-passed capability arguments

Commits: `da5ef4c9c018`, `5a00b4c78b11`.

Decision: treat outgoing stack argument slot addresses as capability-derived
addresses when passing capability arguments beyond the register set.

Why it matters: RV8 `norx` exposed that the common <=8 argument path was correct
while stack-passed capability arguments arrived untagged. The fix is a useful
paper example: preserving tags is not a global property of "using capabilities";
it has to be checked at each ABI boundary.

Limits: this fixes one known ABI path. It argues for the broader capability
authority audit and negative/laundering tests tracked in
`plans/capability-authority-audit.md`.

### Sub-capability aggregate copy correctness

Commit: `161d32974191`.

Decision: fix aggregate `memcpy` lowering for sizes smaller than a capability
slot so scalar copies do not accidentally corrupt adjacent capability-aligned
state.

Why it matters: it is another example of the representation split between
ordinary bytes and out-of-band tags. Small aggregate copies are easy to dismiss
as mundane codegen, but on a capability machine they sit on a security-relevant
boundary.

## Floating point and benchmark reproducibility

### Domain-local soft-float/libm instead of importing a hosted libc

Commits: `a3126729e88d`, `c06b59d36f9b`, `0c259bca869d`,
`0381175a4133`, `de580f57aa3b`, `3ce8537a8b67`, `4ca17bf4f61d`,
`ccc255a0ec84`.

Decision: use a small, explicit soft-float/libm slice for domain benchmarks,
backed by compiler-rt builtins and local fdlibm/newlib-derived routines, rather
than trying to port picolibc/newlib/LLVM libc wholesale at this stage.

Why it matters: this kept the benchmark bring-up focused on compiler/runtime
capability issues instead of a hosted-libc port. It also made dependencies
auditable: every pulled routine is present because a benchmark needs it, and the
exact-oracle tests can compare the same source and rounding policy on host and
target.

Limits: this is a benchmark-runtime strategy, not a general libc strategy.
SQLite or hosted Linux work still needs a principled libc/OS boundary.

### Exact-oracle benchmark method

Commits: `8a4559693112`, `a3126729e88d`, `1daba2f7a007`,
`0c259bca869d`, `89b6d34c4ebe`, `0381175a4133`, `de580f57aa3b`,
`3ce8537a8b67`, `006c2b43e703`, `216396dbafd4`, `18119046df29`.

Decision: when upstream BEEBS verification is missing or unsuitable, keep the
benchmark kernel intact and add a narrow adapted tail that computes an exact
checksum, exact string comparison, or same-source host reference.

Why it matters: the result is a reproducible correctness story rather than
"program ran without fault." This is useful for the paper because it separates
compiler/runtime validation from benchmark-porting convenience.

Limits: host-reference oracles must be kept honest. Each adapted tail should
state what upstream behavior it replaces and why the replacement is equivalent
for the validation target.

### Freestanding libc slice for pure computation

Commits: `006c2b43e703`, `0b7bfb23f6dc`, `18119046df29`.

Decision: implement local `memcpy`/`memmove`/`memset`/string helpers and
benchmark stubs for computation-only workloads, instead of adding a full libc
dependency.

Why it matters: it draws a clean line between pure computation support and OS
services. This makes the absence of hosted libc explicit and lets HostCall stay
reserved for actual host services.

Limits: byte-wise copies strip capability tags. That is safe against forging but
not sufficient for workloads that intentionally copy structs containing
capabilities through byte buffers.

## Regression infrastructure

### Serial aggregate gates first, then opt-in isolated parallelism

Commits: `32c4c6419723`, `ac709d4892b9`, `97fe8317b5a8`,
`dcc9c0cee120`.

Decision: keep individual benchmark wrappers as diagnostic entry points, add
aggregate gates for reproducibility, then add BEEBS parallelism only with
per-benchmark isolated work directories and quiet logs.

Why it matters: this balances developer productivity with the observed QEMU
flake profile. The aggregate scripts are certification gates, while individual
wrappers remain the shortest path for debugging.

Limits: RV8 retries any first failure once; BEEBS retries only structured
pre-benchmark infra flakes. The distinction is worth keeping visible in reports.

### Low-token benchmark certification

Commits: `ac709d4892b9`, `97fe8317b5a8`.

Decision: write child output to per-benchmark logs by default and print compact
pass/fail lines from the aggregate.

Why it matters: agent-driven benchmark certification otherwise burns context on
serial logs. The low-output pattern is now part of the workflow, not just an
agent preference.

Limits: logs remain essential for failures. Reports should cite the log path and
only include relevant tails.

## Suite expansion and deferral decisions

### BEEBS completion by classifying blockers precisely

Commits: `77ce141f5a54`, `8a4559693112`, `a3126729e88d`,
`52033647573d`, `30601e06829d`, `216396dbafd4`, `18119046df29`.

Decision: keep pushing BEEBS once blockers were separated into benchmark
adaptation, soft-float/libm gaps, and real backend bugs.

Why it matters: this produced an 82-wrapper suite without hiding backend
defects. `compress` became a straightforward bring-up; `cubic` drove the
soft-float/libm path; `dtoa` forced capability-global materialization plus
allocator alignment.

Limits: benchmark completion is not equivalent to hosted user-space readiness.

### RV8 C suite complete; C++ bigint deferred

Commits: `5f8692b75bd8`, `97131374070d`, `f06bfc76b203`,
`012b508caeaa`, `0b7bfb23f6dc`, `67b385e17323`, `c02ec1be7b2a`,
`3f286976e454`, `dcc9c0cee120`.

Decision: complete the RV8 C benchmarks and explicitly defer `bigint` because it
is a C++ runtime/STL/toolchain task, not a small benchmark adaptation.

Why it matters: this is a good scope-control example. It avoids turning a
benchmark-suite milestone into an unbounded hosted C++ project.

Limits: the `new` expression backend crash and missing C++ runtime remain real
future work.

## Security and threat-model notes

### Capability table mechanisms are trusted compiler/linker output

Commits: `30601e06829d`, `e7d5b87656e1`.

Decision: startup trusts the linker-bounded `.capstone_cap_init` table and calls
each entry before `domain_main`.

Why it matters: the mechanism is correct for trusted object files, but the table
is not a sandbox boundary. An adversarial linked object can add an entry that
calls arbitrary domain code, which is not a new authority if object files are
already trusted, but must be stated clearly in the threat model.

Limits: add linker assertions and documentation if this becomes part of a
malicious-object-input story.

### Current central safety gap: broad bounds

Related artifact: `capstone/agent-handoff/design/capability-provenance-threat-model.md`
when committed.

Decision: treat object-bound tightening as a separate research track, not as
something already delivered by the recent benchmark work.

Why it matters: current generated capabilities preserve tags but often inherit
broad roots such as `gp`, stack, or allocator arenas. The paper must not claim
object-granularity spatial safety until bounds are tightened and tested.

Limits: this is a research/audit finding, not a fix in the audited commit range.

### Tag-granularity overlap: a scalar store inside a live capability's 16-byte granule silently strips its tag

Status: research finding (2026-07-02), fix pending. Full diagnosis trail:
`capstone/agent-handoff/history/02-07-2026_00-00-00_sqlite-gap5-fix-and-gap6-investigation.md`
(SQLite gap 6). Not yet a commit — recorded here because the *class* is
paper-worthy independently of the eventual fix.

Finding: the hardware tag map is **16-byte-granular** — any scalar write anywhere
inside a capability's 16-byte storage clears that capability's tag (correct
capability semantics: a partial overwrite is a forge attempt). This makes storage
**layout** a provenance-correctness property, not just an ABI convenience. If the
compiler ever places a scalar object (a spill slot, a coalesced local, a struct
sub-field) in the same 16-byte granule as a still-live capability, a store to the
scalar de-tags the capability with no diagnostic; a later `ldc` returns it
untagged and the first dereference faults far from the cause.

Evidence: SQLite `sqlite3DeleteTable` faulted on an untagged `Table*`
(`0x102247f50`, a MEMSYS5 allocation). The allocator returned it **tagged**;
aggregate copy, varargs, and pointer↔int round-trip were all ruled out
(tag-preserving). Static disassembly of the faulting function (`-g` build,
symbolized 2026-07-02) shows DeleteTable's own `pTable` slot is **clean** —
`stc` in, scalar *load* for the `if(!pTable)` null check (loads don't clear),
`ldc` out — so the pointer **arrives untagged from the caller**. The loss is
therefore **upstream**, on the CREATE→schema-hash-store→DROP-retrieve path: the
`Table*` is parked in `pSchema->tblHash` (a `HashElem` in `sqlite_heap`) and
retrieved by `sqlite3UnlinkAndDeleteTable`. The exact clearing store is not yet
pinned; a value-only detector over `0x102247f50` is too loose (the value is
pervasive — three sampled `TAG-ST-CLR` pcs symbolized to unrelated functions),
so the remaining step is a **storage-slot-keyed** trace of the HashElem granule.
The candidate mechanism remains a **memory-layout × 16-byte-tag-granularity
interaction** (a scalar packed into the same 16-byte granule as the stored
capability), now to be confirmed against the specific HashElem slot rather than a
stack frame.

Why it matters: this is a distinct tag-loss class from the earlier backend bugs
(`va_list` scalar path, stack-passed cap args, sub-capability aggregate copy),
which were all *value-motion* bugs — a capability travelling a scalar path. This
one is a *storage-aliasing* bug: the capability never moves, but a neighbour
overwrites part of its granule. For the paper it argues that capability-tag
preservation is not only an ABI/lowering obligation but also a **stack/struct
layout invariant** the backend must maintain: no scalar storage may share a
16-byte granule with a live tagged capability.

Limits: the specific causal slot is not yet pinned to LLVM source/MIR — the next
step is `-g` + `llvm-symbolizer` on the `TAG-ST-CLR` pcs and confirming the clear
hits the exact slot `sqlite3DeleteTable` reloads from (the value is pervasive, so
incidental clears must be excluded). The fix is substantial backend work and will
get its own proposal doc before implementation.
