# Virtual Capstone review and landing plan

Prepare a dependency stack of fresh review branches from current shared targets.
Keep the existing working branches as provenance. Do not replace dev with the
virtual tree: newer physical-runtime and application work exists there, and the
qualified virtual binary has not integrated all current shared QEMU fixes.

The direction is supervised **virtual C mode**, with a trusted kernel and
adapter, Linux scheduling/VM, and one lifetime namespace per address space.
The earlier direct protected-U Linux experiment remains a separate lane.

## Audited bases

The [audit manifest](virtual-capstone-pr-audit.json) records full hashes, all
41 main-repo and 26 QEMU feature commits, paths, sibling overlap, submodule pins
and dry-merge conflicts. This snapshot includes pthreads and memcached.

| Repository / ref | Snapshot | Review role |
|---|---|---|
| llvm-capstone feature | `6bf819677207` | Complete current virtual application lane |
| llvm-capstone dev | `4ea8ea9f6fa3` | Landing target; 269 target-only versus 41 feature-only commits |
| llvm-capstone main | `88618641986d` | Older release; promote after integration into dev |
| delegation-threads | `92d21b84324b` | Source of reused musl layouts and host signal files |
| memcached-app | `c839e7294e23` | Imported physical baseline; older than current dev |
| virtual-addressing | `6b5779a09072` | Sibling comparison, not a landing target |
| virtual-capstone-prototype | `55708215b540` | Ancestor containing protected U and guest tables |
| virtual-capstone-runtime-parity | `b7c0ad843f87` | Ancestor of the C-mode runtime |
| QEMU feature | `6bf04634a2c2` | Qualified virtual processor model |
| QEMU c128-qemu-merge | `b8f08e599330` | Target; nine target-only versus 26 feature-only commits, including merges |
| Academic ISA master | `ac7f32910048` | Spec base; newer than the main repo's `eadaf872705e` pin |
| Linux qualified source | `a62e18f2b9aa` | No additional kernel-core change for the C-mode adapter |

Use three-dot diffs for historical feature work; use two-dot comparisons to
inspect final behavior on selected paths. Include submodules explicitly with
`--ignore-submodules=none`: the local default hides them. Only QEMU changes as
a feature-side gitlink. Preserve dev's newer RTL `776d9d85900e` and Buildroot
`8fd1ea1249a9` pins. The system pin is unchanged. No RTL, Buildroot, system,
Linux-core or firmware PR is required by this lane. The draft spec branch does
not require an immediate superproject spec-pin bump.

## Findings that change the plan

1. **The existing two review branches are slices, not dev-ready PRs.**
   `pr/virtual-linux-posix` is one commit on `6fb2df614ca4`;
   `pr/virtual-memcached` is one commit on that pthread branch. Against dev,
   either would expose historical prerequisites too. Keep them as provenance.
2. **Integration needs explicit conflict resolution.** A dry `git merge-tree`
   reports 31 conflicting main-repo paths, including the QEMU gitlink, and two
   QEMU files: `cap.h` and `op_helper.c`. The manifest lists them. Automatic
   merges on other paths still need semantic review and execution gates.
3. **Dev's memcached is ahead of the imported baseline.** The virtual delta
   against memcached-app is only eight files, 676 additions and seven deletions.
   Dev additionally has slab/cache Sublet hooks, corpus-in-server probes,
   bipbuffer and later spatial fixtures. Preserve these and adapt the virtual
   recipes/runner to them. The current 53-check virtual result covers the older
   baseline, not all of current dev. Its slab-neighbor/reuse gaps do not mean
   the newer physical slab-Sublet arm lacks protection.
4. **tshark also has evidence drift.** Dev contains newer wmem chunk/corpus
   fixtures and expectation rows. Preserve them and its dependency fixes when
   adding the virtual heap adapter; do not overwrite its safety source or
   expectations with the older version. Keep allocator-profile differences explicit.
5. **Reused code needs attribution of provenance, not duplicate review.**
   Musl patches 0004/0005 and `runtime/linux/signals.{c,h}` are byte-identical
   to delegation-threads at the audited refs. Shared delegate, level0, TLS and
   atomic paths require hunk-level integration with dev's physical support.
6. **QEMU target-side fixes must survive.** Preserve dead-seal status,
   supervised C-mode confinement, sealed-return slot confinement and
   overflow-safe sealed-return windows, plus instrumentation. The virtual
   lane's subtraction-based `cap_in_bounds` does not replace the separate
   sealed-return window fix. Two textual conflicts are not the whole audit.
7. **The ISA needs a scoped profile.** Academic reference, QEMU and RTL differ.
   CSCHECKR/W are not implemented; fault data differ from the academic text;
   compressed bounds with tag-bit-only storage remain an open conformance issue.

## Proposed PR stack

These are extraction boundaries, not claims that historical commits cherry-pick
independently. Each PR must build and carry its relevant gate. Squash each
complete logical change when landing on its shared target; do not rewrite
published provenance branches.

| ID / proposed title | Repository and base | Scope and acceptance |
|---|---|---|
| S1: Document the trusted-OS virtual execution profile | academic-spec master | Draft ownership/ISA/ABI appendix, implemented behavior and open freeze decisions; HTML/PDF render. Ready for review independently. |
| Q1: Enforce protected capability access and physical tags | QEMU c128-qemu-merge | Physical tags/kernel-store invalidation, rights/bounds, trusted S-mode slot transfers, consuming Q-12, explicit fault data/delegation; protected and physical regressions. Retain experimental opt-in. |
| Q2: Select bounded guest lifetime tables and trusted minting | QEMU Q1 | Shared revocation algorithm/model, fail-closed IDs, guest records, three CSRs, CSMINT; initially monotone IDs. Independent-root, capacity, corruption and mint-preflight gates. |
| Q3: Enter supervised virtual C contexts | QEMU Q2 | CSRUNV, user-PTE translation, execution-time PCC, instruction policy, service/fault/quantum events, CSRETIRE; virtual gate including cached-PCC and exact retries. |
| Q4: Recycle IDs after a stopped-namespace sweep | QEMU Q3 | Free-list header, node pressure, CSRUNV action 3, stale memory/register tag removal and dead-PCC pinning; inventory/refusal, reuse and sustained-allocation gates. |
| C1: Support readable image-gp constants | llvm-capstone dev | Two backend files and cap-image-gp-pool test; retain captable refusal. Include deterministic drift-helper output as validation infrastructure. |
| D1: Preserve caller authority across delegated services | llvm-capstone C1 | Net buffer-check/snapshot changes, integrated with dev; bounds, short/error copies and physical I/O regressions. Omit equivalent target hunks. |
| R1: Run virtual C applications through a Linux adapter | llvm-capstone D1; merged Q4 pin | Module ownership, launcher/CRT, SDK and common runners, events and explicit same-mm contexts. SQLite/mruby, independent processes, caller restoration and thread-exit gates. |
| R2: Provide virtual libc mappings and a shared revoking heap | llvm-capstone R1 | VM services, anonymous mappings/protection, growing heap/metadata and nested-allocator block loans; VM, spatial/temporal, same-VA reuse, shared-heap and recycling gates. |
| R3: Support musl pthreads in virtual applications | llvm-capstone R2 | Adapt existing pthread commit: TLS, workers, per-call transports, native-VA futexes, safe exit, ABI v3; nine-check gate, v2 compatibility, both buffer mutation controls. |
| A1: Build and qualify virtual Perl | llvm-capstone R2 | Fresh source recipe, retained pointer-width patches/configuration limits, smoke and corpus gates. |
| A2a: Build and qualify virtual CPython | llvm-capstone R2 | Normal/inner workloads; CPY_SUBLET_MODE=1 for temporal inner protection. |
| A2b: Build and qualify virtual PostgreSQL | llvm-capstone R2 | Single-user normal/inner workloads, including node-budget behavior. |
| A2c: Build and qualify virtual FFmpeg | llvm-capstone R2 | Decoder profile, block loans and unchanged applicable safety predictions. |
| A2d: Build and qualify virtual tshark | llvm-capstone R2 | Offline/wmem profile on dev's newer chunk/corpus fixtures and dependency fixes. |
| A3: Run the complete server on virtual Capstone | llvm-capstone R3 | Virtual memcached delta atop current dev; preserve all physical arms. Native protocol/workers/TERM/USR1 and current applicable corpus matrix. |

Review S1 alongside Q1--Q4; agree the profile before freezing allocations.
C1/D1 can proceed while QEMU is reviewed. Merge and qualify QEMU first, then
update its superproject pin in R1. R2/R3 follow. The five A1/A2 leaf PRs are
independent after R2; A3 depends on R3. Release to main follows integrated dev.

Common SDK/runner infrastructure and block-loan primitives belong in R1/R2,
so leaf app PRs do not repeatedly modify shared runtime files. Aggregate
all-port evidence belongs with the final integration gate after all referenced
recipes exist. MicroPython remains outside this migration.

## Extraction map

| Historical source | Landing ownership |
|---|---|
| QEMU db53c89ad0 through tag/access fixes; f153d90d29, d2da1a852c, f5f0ed887e, cbd3764a64 | Q1, adapted on current physical-supervisor fixes |
| QEMU e9413fe3d2, 3a411e6379, dc8878185d | Q2; model, guest table and mint representability evidence |
| QEMU 5adb911043, 49328837b0 | Q3; execution and continuation |
| QEMU 8ad8d28158 and evidence correction 6bf04634a2 | Q4; the complete sweep/reuse protocol |
| main 4715ec129b60, 923cc8c235ae | D1; check target equivalence first |
| main bd13d9348b75 through da36ed0e1d8e | R1; extract compiler hunks into C1; retain final same-mm lifetime fixes |
| main 320ba350f278 and common heap/build/runner hunks in 6fb2df614ca4 | R2; source-build pieces go to R1 where required |
| main 35cb6a338467 | R3; preserve physical-path changes already in dev |
| main e2777455c4f1 and app-specific hunks in 6fb2df614ca4 | A1 and A2a--A2d |
| main 6bf819677207 | A3 virtual delta, without reimporting old physical memcached |

The initial protected-U probes, Linux entry patch, selector history and partial
register-spill qualification stay in their experimental lane. Keep the trusted
S-mode LDC/STC support: the current module uses it in a short IRQ/preemption-
excluded grant-conversion sequence and clears its tagged temporary afterwards.
Retained QEMU
U-mode support can keep its own tests; R1 must not accidentally require that
Linux patch. The physical legacy monitor remains supported.

## Review contract

Every PR description needs a concrete before/after behavior, trusted components,
dependency PRs/pins, source hashes and its gate command/results. Explain ABI
breaks (VM-service v2/v3), configuration limits and workload scope. Runtime
service numbers are not CPU opcodes. Consolidate durable documentation with
its owning change; do not replicate intermediate state logs.

For QEMU, inspect complete preflight/commit paths, root isolation and tag
lifecycle. Preserve the same-physical-granule consuming-load check and the
execution-time cached-PCC check. Recycling needs separate review of complete
page inventories and saved contexts; trusting Linux leaves those correctness
obligations intact.

For R1/R2, review module/capstone_vm.c, exec.c, thread.c, mapping.c, heap.c,
wire ABI and public header with the lifecycle gates. Roots belong to the mm;
registers/PCC/TLS belong to contexts. Cleanup, retirement and reuse must
preserve this distinction. For R3 retain both the per-invocation epoll conversion
control and blocked-receive host-view/iovec control. Private syscall transports
alone did not remove both shared-buffer races. Reused signal support does not
prove full POSIX signal/cancellation coverage.

App PRs must run against current target recipes, patches and predictions.
Record old-baseline results separately; do not change predictions to make
virtual tests pass. The current 53 memcached, nine pthread, 46 runtime and
recorded QEMU gates are provenance, not measurements of the planned merge.

The final gate rebuilds QEMU/compiler, runs physical/protected/virtual CPU
tests, VM/thread/security gates and all eight requested application profiles,
with native controls and current applicable corpus arms. This documentation
review reruns no runtime workload. Source inspection, dry merges and spec
rendering do not substitute for integrated execution tests.

## Prepared branches

- `virtual-capstone-review-plan` contains this proposal and audit on the
  qualified `6bf819677207` baseline. It is a planning branch, not an
  implementation PR against dev.
- The local `virtual-capstone-isa` branch in
  [academic-spec](https://github.com/project-starch/capstone-academic-spec)
  starts at master `ac7f32910048`. Commit `6e05ba6` specifies ownership,
  exact CSRs/opcodes, frames/events, Q-12, revocation, collection, and differences
  from the academic ISA. Full HTML/PDF rendering passes with warnings treated
  as failures. Publication to that repository is blocked by GitHub write
  permissions (HTTP 403); there is no published spec branch yet.
  The [transferable spec patch](virtual-capstone-isa.patch) contains the exact
  three-file diff without commit identity metadata. In an academic-spec
  checkout at `ac7f32910048`, create a review branch, then run
  `git apply --check /path/to/virtual-capstone-isa.patch` followed by
  `git apply /path/to/virtual-capstone-isa.patch`. It changes only the spec
  README, main include list and the new profile appendix.

Existing working and stacked review branches remain intact. Q1--A3 are proposed
implementation slices; none has been silently rebased, merged or claimed tested
against current dev.
