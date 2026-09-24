# Current Capstone state

Minimal snapshot. Read first in every session.

## 2026-09-24 — R-35's fix is on silicon, and the probe that exposed R-35 no longer reproduces it

**ON SILICON (2026-09-24, caplifive_r35_4ad0df694.bit):** the probe that exposed R-35 no longer reproduces it. The same image that read the current occupant's live data through a revoked alias on 054cea69b (is_live_data=1) now traps with cause 25 on the stale read of leaf[0]. A live alias to the same leaf, at the same address, commits without a trap, as do ~43k earlier live accesses, and 17 of 17 ladder rungs return their pre-flash values. N=1 per arm. NOT shown on silicon: WHY the stale read was denied -- cause 25 cannot separate observed revocation from the cache's deny-on-miss, which is the expected route for an id reissued thousands of times; which probe age trapped (k=0 or k=21648); the stale write; and false-deny rates under SQLite. Result lines and limits: the R-35 folder's `results/board-4ad0df694.result-lines.txt`.

**Next for R-35:** a SQLite workload on this image, which is the false-deny test at scale. If the lead wants the denial attributed, a k=43295-only variant paired with a long-idle live alias separates revocation from residency. **R-42** (I-cache killed-miss, performance): fix at capstone-ariane `6cbdaeeb4`, one commit on `4ad0df694`, validated in simulation; synthesis running (authorized by the lead 2026-09-24).

## 2026-09-24 (earlier) — R-35's fix is synthesized and timing-clean at `4ad0df694`: a reflash candidate, not yet on silicon — **SUPERSEDED by the section above**

**R-35 FIX IS SYNTHESIZED, TIMING-CLEAN, AND A REFLASH CANDIDATE (2026-09-24).** `capstone-ariane` **`4ad0df694`** routes at **WNS -8.341, 0.034 ns from the flashed base's -8.307**, with 51.70 % failing endpoints against base's 51.76 %. All five pre-registered synthesis predictions pass. Bitstream sha256 `8db73f8e20244438a2663fef070202e95dde29fe5b1d957b9804a7babf60382c`. *(Flashed since; see the section above.)* Full readings: the R-35 folder's `results/synth-4ad0df694.result-lines.txt`.

**What fixed the timing, in two parts** — `f83fe9342`'s −27.665 had two causes. The rev-node's *combinational* write-request selector drove the cache's write decode; `a87a24a59` registers the fill taps. And the cache's footprint crowded a congested region; `4ad0df694` moves the tag array into distributed RAM. Paths that went *past* the cache recovered ~18.5 ns untouched, which measures the congestion rather than inferring it.

**The correct revocation check turned out to be essentially free.** An earlier claim that it must cost about what Stage 0 cost — because the base was cheap *by being vacuous* — is refuted: Stage 0's 4.593 ns was its implementation, not the price of correctness.

**Next:** put the `.bit` on the console's server-side BITSTREAM store (GUI Bitstream Manager or the board owner — our driver would file it as a boot image), verify its hash, flash (authorized by the lead), and run the folder's board probe. The board lane takes the window after, for its ladder refresh — which doubles as the false-deny test at scale.

## 2026-09-23 — R-35 is fixed in simulation at `f83fe9342`; it is NOT yet deployable, and neither synthesized hash is — **SUPERSEDED 2026-09-24 by the section above**

**Current fix: `capstone-ariane` commit `f83fe9342`, branch `r35-m1-revnode-cache`, pushed.** `f83fe9342` WAS SYNTHESIZED 2026-09-23 AND IS NOT DEPLOYABLE. Area: fixed -- 177,669 post-synth LUTs, -17,423 against the crossbar build with -111 FFs, i.e. the crossbar rewritten as a register file and nothing else. Slack: REFUTED -- routed WNS -27.665, the worst this design has produced, against a pre-registered prediction of roughly -12.9. The worst path is SHALLOWER than Stage 0's (94 vs 121 logic levels, less logic delay) but carries +15.45 ns of ROUTE delay: congestion, absent from every earlier build. Traced at pin level by the synthesis lane: D-cache read-port grant (fan-out 88) -> the rev-node's memory-channel write-request selector, a deep combinational Anvil mux (fan-out 70) -> the cache's WRITE decode. The fix hung a 256-entry write decode off that selector. The read mux is not on the path. Next: register the fill taps, so the decode sees flops.

What follows was written before that result: It replaces the LSU's single core-wide revnode tracker with a tagged 4-way ×
64-set positive validity cache, filled **only** by two passive taps on the rev-node unit's own
node-memory traffic, so an access is allowed only if its exact 30-bit `(generation, index)` is resident
and was last seen live. Acceptance fixture (`r35-rotate-stale.S`): exactly **7 traps**, the three
revoked accesses trap 25, both live-alias controls still return data. Lint at the committed baseline.

**Neither synthesized hash is a reflash candidate:**

| hash | what it is | routed WNS | failing endpoints | routed LUTs |
|---|---|---|---|---|
| `054cea69b` | flashed base | −8.307 | 51.76 % | 168,757 |
| `247b76896` | Stage 0 (tracker reorder) | **−12.900** | 57.25 % | 169,953 |
| `079dc720a` | first cache | **−14.415** | 60.82 % | **192,642 = 94.53 %** |

**The cheap-looking edit was the expensive one.** Stage 0 — 32 lines, zero new signal declarations —
cost **4.593 ns**; the entire first cache cost 1.515 ns more. A before-audit localized Stage 0's cost to
its **LSU** half — *audited, not measured: no build has separated the two halves Stage 0 changed* (the post-adopt value fed `cap_exception` combinationally); every CPMP consumer reads
the registered value, so the CPMP half is flop-to-flop into 16 endpoints. Loops stayed at 1 on every
build: these are path-depth costs, not new cycles. The first cache's LUTs came from describing a
crossbar — a `_d`/`_q` pair rebuilt by dynamic index every cycle, about 90 LUTs per entry.

**All of these costs were invisible to lint:** nine counters at exact baseline and `UNOPTFLAT` unmoved at
40, on every hash.

**The register-file rebuild (`6ee277cc3`) introduced an authority escape — R-35's own class — closed at
`f83fe9342`.** An adversarial after-audit found it with a microtest built from verbatim extracts plus
mutants as positive controls. **The acceptance fixture returned exactly 7 traps both before and after the
fix**: it cannot see same-cycle coincidences. Do not treat a passing fixture as evidence about paths it
does not reach.

**Next:** synthesize `f83fe9342` against pre-registered area and slack predictions (see the plan). The
reflash, and the board run after it, are the lead's call.

**Do NOT:** flash `079dc720a` — at −14.4 ns and 60.82 % failing endpoints a flake would be
indistinguishable from the fix not working. Do NOT revert the CPMP half of Stage 0 to compare the
registered value as a timing "fix" — it admits a persistent false ALLOW (entry adopts Y while a
broadcast names Y; the compare against the stale X misses; the entry vouches for a dead Y forever).

## 2026-09-22 — R-35 registered, and Stage 0 of its fix is ready for synthesis — **SUPERSEDED 2026-09-23: Stage 0 was synthesized and is a 4.593 ns regression, and the "Stage 1+2 / Stage C refill" framing below was replaced by the positive cache. See above.**

**R-35 — a REVOKED capability still reads and writes the storage its object gave up, and the access
does not trap.** Root-caused at `054cea69b` to `load_store_unit.sv`'s single core-wide revnode
tracker, whose adopt arm takes any unseen id as valid without asking the rev-node unit. Reproduced
**two-sided in RTL simulation** with a single-variable arm pair, a witnessed revoke and the faulting
unit attributed by a control that demonstrably fires — strictly better evidence than the board
capture, which sits on a bitstream whose timing does not close. Folder:
`capstone/tests/fpga-repros/R35-revoked-reference-retains-authority/`.

**Now in the registry**, which stopped at R-34 before today: **R-35** (the defect), **R-37** (the
trackers' invalidate/adopt mis-ordering), **R-38** (the CPMP tracker's pre-adopt compare, a permanent
false deny), **R-39** (unguarded index-0 invalidation broadcasts, believed harmless on a stated
invariant), **R-40** (no authorization check on genesis region ids 0-2 — **materially corrected**
after an auditor refuted its first mechanism), **R-41** (the type guard's early return drops the CPMP
entry it read; unproven). **R-36 is a stub**: the number was drafted 2026-09-21 and withdrawn before
filing, and `history/` references it.

**Stage 0 of the fix is implemented, validated and awaiting synthesis.**
`capstone-ariane` branch **`r37-stage0-tracker-ordering`**, commit **`247b76896`**, pushed. It moves
each tracker's invalidate **after** the adopt and compares the **post-adopt** value — the shape
`commit_stage` has always had.

* **Lint: identical to the committed baseline on all nine counters** — LATCH 52 / MULTIDRIVEN 3 /
  ALWCOMBORDER 0 / COMBDLY 0 / UNOPTFLAT **40** / BLKSEQ 2 / UNDRIVEN 25 / UNUSEDSIGNAL 736 /
  ANVIL_UNOPTFLAT 0, over 6,511 lint lines. Two files, 32 insertions, no new signal declarations.
* **Full directed sweep, matched pair, seed pinned both sides: ZERO status changes and ZERO cycle
  drift** (66 PASS / 26 TIMEOUT / 3 NOBUILD / 0 FAILED on each side; 92 comparable tests with
  bit-identical cycle counts). The baseline arm reverted only the two files, in the same worktree.
* **Both halves validated functionally.** The M-mode reproducer is bit-identical, and the CPMP half
  was validated separately on `r12-recl-cpmp.S` — the only fixture that reaches the CPMP gate, since
  that check is gated on privilege *not* being M and the M-mode fixture is structurally blind to it.
  Pre-registered `0 1 1 0 1 0 1` / total 4, matched exactly.

**IT STILL NEEDS SYNTHESIS, and "signal-neutral" was wrong.** Retargeting a compare from `_q` to `_d`
swaps a flip-flop output for a **combinational** node inside two modules that are both in the standing
`UNOPTFLAT` loop set, which no lint counter can see. Per the project rule about feeding a signal into
a cone that already carries a loop, this goes to synthesis before a board or another lane.
**`054cea69b` is NOT an ancestor of the submodule's working-tree HEAD**, so any hash handed on must
name its line.

**Stage 1+2 (the validation broadcast and registry-gated adopt) is NOT started, and its price went
up.** A trap-path enumeration established that the LSU's plain-dereference working set is **~20 ids
and rising**, so registry-capacity eviction can miss *any* id — including the monitor trap stack's,
whose 30-store prologue would then nest silently. So **Stage C (refill) is mandatory rather than
likely**, and it is either a hardware query with a stall (crosses the combinational ring) or a
retryable cause 25 (an ABI commitment). That is a project decision, not a lane's.

**Two instrument facts from today that outlive this work**, both now in the `rtl-sim` skill:
`cva6.py`'s `--sv_seed` defaults to a **fresh random value**, so "zero cycle drift" is unassertable
unless it is pinned on both sides; and `cva6.py` returns **0 for a TIMEOUT as well as a PASS**, so an
exit-status tally cannot classify a sweep.

## 2026-09-18 — Whisper ggml context component

`capstone/ports/whisper/ggml-context/` ports whisper.cpp 1.9.4's real context
allocator through the shared template. A native tiny.en recording contains
60,179 events and 58,192 object allocations; stock and instrumented transcripts
match. Native allocation layout matches both the extracted reference and the
ordinary full ggml library. The recording completes in spatial and Sublet QEMU.
Borrowed-buffer graph objects survive descriptor destruction; reset, owned free
and exclusive owner rebind are distinct epoch boundaries. The README documents
that rebind contract, capability-header capacity adjustment and fixed backing
budget. This is allocator replay, not protected inference or FPGA measurement.
## 2026-09-18 — Opt-in generic client-fault recovery (QEMU)

The [shared runtime](../../runtime/domain-faults.md) provides a domain build
helper, cooperative fault return/quarantine, and a Linux process-termination
policy. Its standalone tests require no PostgreSQL or Sublet allocator sources.
Enable `CAPSTONE_DOMAIN_FAULT_RECOVERY` only with the matching trap-delivery QEMU.
This is launcher-chosen SIGSEGV termination, not monitor-enforced containment,
complete resource reclamation, or a new FPGA result. Allocator integration is
reviewed separately; existing ports are not enabled automatically.
Port navigation and pending integration: [component catalog](../../ports/README.md)
and [integration plan](../plans/port-stack-integration.md). The shared runtime's
missing `include/sublet/sublet.h` is restored from the identical port-branch
header. This repairs a missing build input; the dated silicon results below
and the pending fault-recovery validation remain separate.
## 2026-09-19 — Experimental PoisonCap pymalloc port

The extracted CPython 3.13.7 allocator now has a trusted PoisonCap backend,
reusing the three existing upstream patches and the FFmpeg platform. The full
33-process QEMU suite passes: ABI/platform controls, linked example, five API
checks, nine paired lifetime cases and two native-recording replays. Both
modes process all 115 events and match the native logical oracle; payload
preservation is checked inside the guest. The small complete recording is not
the existing larger workload or the defect corpus.

An initial failed replay exposed stored poison capabilities remaining after
`cclearpoison`, causing a later sweep to revoke a fresh unwritten allocation.
The adapter now overwrites those remnants and counts the additional writes;
a targeted regression and the complete suite pass. The protected recording
uses 63 sweeps and 80,336 bytes each of poison/clear/zero work. Snapshot copy
traffic is zero on this recording; separate in-place realloc controls exercise
it. Both modes report 5,275,200 bytes of private metadata high-water, including
the same authority-record layout and replay scratch. These are not total
memory overhead or hardware timing measurements.

Automatic libc revocation remains explicitly off while adapter sweeps remain
on, using the documented platform workaround. Native tests and all four
backend builds pass. This is not whole-interpreter protection or isolation of
hostile nested managers. [Pilot and provenance](../../ports/cpython/pymalloc/results/20260919-poisoncap/README.md),
[build/link/run guide](../../ports/cpython/pymalloc/host/cheribsd/poisoncap/README.md).

## 2026-09-19 — Experimental PoisonCap FFmpeg port

The published PoisonCap compiler, QEMU and matching CheriBSD kernel/userspace
are reconstructed with pinned sources. The FFmpeg allocator library, direct
example and per-lease adapter build. Seven platform controls pass; ten protected
pool cases pass in a separate fresh guest, including persistent RefStruct state
and stale access after reuse. A 2,379-event native recording matches its complete
event oracle. The first adapter snapshots payloads before poisoning and sweeps
before reuse, with its storage and copy costs counted separately.

The full suite passes all 29 processes when the automatic guest libc-revocation
default is disabled before SSH starts; explicit adapter revocation remains
active. Three successful replays have identical output and counters. Preserving
the guest default instead produces a captured kernel `share->excl` panic in
longer suites, including a spatial-only arm. The explicit guest configuration
is a workaround, not a kernel fix or a Capstone/PoisonCap performance ranking.
[Pilot, failed attempts and scope](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-poisoncap-pilot/README.md).

The [backend regression](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-poisoncap-pilot/README.md#backend-regression-verification)
rebuilds all four FFmpeg configurations, passes 4 native and 23 shared Python
tests, and repeats the full 29-process PoisonCap suite with an identical replay
report. The spatial CheriBSD replay and its five controls also pass.

## 2026-09-19 — Four CheriBSD allocator libraries and examples

FFmpeg buffer pools, PostgreSQL memory contexts, CPython pymalloc and Whisper
ggml contexts now share a CheriBSD purecap toolchain, build/run scripts and
CMake library targets. Standalone examples and separately supplied client
sources link and run in QEMU. The suite checks the ABI/runtime policy and an
exact bounds fault before running allocator programs. FFmpeg, CPython and
ggml native-recording replays match their logical native oracles; PostgreSQL
passes its four-manager fixture. Native regression tests pass, and all four
Capstone domain configurations still build. These are capability-compatible
component ports with explicitly documented boundaries, not automatic inner
temporal protection in CheriBSD.
[Build/link/run guide](../../ports/common/host/cheribsd/README.md).

## 2026-09-19 — Scattered aliases and ancestor revocation

The synthetic A1 fixture passes 44 QEMU executions: 20 stale read/write
attempts fault after parent revocation, 20 matched no-revoke attempts complete,
and four valid-authority controls complete. Five alias locations are covered
(global, heap object, linked list, independent sibling pool, register), both
immediately after revocation and after same-address reuse. Disassembly confirms
the register alias stays in a register across revocation without calls or
spills. New authority and the unaffected sibling remain usable. This is
functional Capstone evidence, not protected decoding, hostile-manager domain
isolation, a performance measurement or a measured competitor disadvantage.
[Matrix, protocol and provenance](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-alias-scatter/README.md).

## 2026-09-19 — CHERI spatial arena comparison

Nine CHERI spatial replays match the native recordings, with three identical
repetitions per workload. Five companion controls distinguish bounds faults
from ordinary stale pool accesses. The
[three-arm export and plots](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-cheri/README.md)
contain Capstone spatial, Capstone Sublet and CHERI spatial only. Payload
padding and static storage are separate observations; the arms do not provide
equivalent lifetime guarantees or a complete protection-memory ledger.

## 2026-09-19 — Paired FFmpeg replay measurements

The measurement worktree adds a reproducible native-to-QEMU comparison on
three FFmpeg recordings. Eighteen accepted spatial/Sublet points match the
native event sequences, with three bit-identical repeats per workload/arm.
Six companion lifetime controls and four native CTests pass. Failed runner
attempts are retained. These are allocator observations and resource counters,
not application timing or a complete protection-memory ledger. See the
[result bundle](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-replay/README.md)
and [measurement plan](../plans/replay-memory-measurements.md).

Trace-development branch: the current port/runtime/corpus PR heads are combined
in `integration/2-trace-ports`. The [shared trace tooling](../../ports/common/host/port_trace/README.md)
reads all four existing formats and supplies staged-input validation and
versioned result metadata. This is host tooling; allocator replay semantics
and previous QEMU/silicon result identities are unchanged.

Port navigation and pending integration: [component catalog](../../ports/README.md)
and [integration plan](../plans/port-stack-integration.md). The shared runtime's
missing `include/sublet/sublet.h` is restored from the identical port-branch
header. This repairs a missing build input; the dated silicon results below
and the pending fault-recovery validation remain separate.
## 2026-09-18 — PostgreSQL's four Sublet allocator ports

The canonical `ports/postgres/memory-contexts` CMake component ports AllocSet,
Generation, Slab and Bump at PostgreSQL 17.0. Generation/Slab revoke individual
allocations; all four support context reset/delete. Bump retains bulk-only
lifetime semantics. Upstream block policies remain in the versioned patches,
with out-of-band context/block metadata and allocation authority from Sublet.
Native tests, mixed-manager replay and paired exact-access QEMU fixtures are
documented in the component README. Debug manager layouts are explicitly
refused. The original shell builds remain AllocSet-only, and protected
consumer-defect reproduction is still a separate milestone.

Runnable clients are in `ports/postgres/memory-contexts/examples/`: an AllocSet
buffer, Generation message queue, Slab job table and Bump request scratch.
Each source is shared by native/spatial/Sublet executables. All four pass in
all three builds; the examples README gives commands and the client interface.

## 2026-09-18 — PostgreSQL 17.0 pin consistency (before the additional ports)

The PostgreSQL component, original shell builds and native defect corpus now
read one `memory-contexts/upstream.json` pin: 17.0. Five native and four QEMU
CTests pass; both original domain scripts compile and the native defect
reproduces. Historical 17.5 profiles retain their original provenance. The
shared Sublet runtime header required by the CMake ports is restored, with a
configure-time completeness guard. See the component README for commands and
scope: AllocSet is protected; the consumer reproducer's Capstone arm remains
future work.


## 2026-09-18 — Opt-in generic client-fault recovery (QEMU)

The [shared runtime](../../runtime/domain-faults.md) provides a domain build
helper, cooperative fault return/quarantine, and a Linux process-termination
policy. Its standalone tests require no PostgreSQL or Sublet allocator sources.
Enable `CAPSTONE_DOMAIN_FAULT_RECOVERY` only with the matching trap-delivery QEMU.
This is launcher-chosen SIGSEGV termination, not monitor-enforced containment,
complete resource reclamation, or a new FPGA result. Allocator integration is
reviewed separately; existing ports are not enabled automatically.

## 2026-09-18 — Whisper ggml context component

`capstone/ports/whisper/ggml-context/` ports whisper.cpp 1.9.4's real context
allocator through the shared template. A native tiny.en recording contains
60,179 events and 58,192 object allocations; stock and instrumented transcripts
match. Native allocation layout matches both the extracted reference and the
ordinary full ggml library. The recording completes in spatial and Sublet QEMU.
Borrowed-buffer graph objects survive descriptor destruction; reset, owned free
and exclusive owner rebind are distinct epoch boundaries. The README documents
that rebind contract, capability-header capacity adjustment and fixed backing
budget. This is allocator replay, not protected inference or FPGA measurement.

## 2026-09-17 (evening) — CURRENT

> **THE BOARD WAS REFLASHED AND CAPABILITY EXCEPTION DELIVERY IS NOW LIVE ON SILICON.** The reclaimer
> build `054cea69b` is resident, carrying the revoke-walk splice, the node reclaimer **and R-34 plus
> R-24** — the last of those found by reading the build's contents rather than its name, two hours
> before the flash. **Every board result recorded before this reflash is on the old silicon and is
> stale under the standing rule**; re-check before relying on one.
>
> **The monitor fix was a prerequisite and it held.** With delivery live, the old monitor's integer
> add on the stack capability would have faulted inside its own trap handler at every `rdtime` and the
> boot would have looked like a bad bitstream. First boot after the merge: control `retval=4` twice,
> **zero refused-or-trap lines**, all three invocations returned. The pin is `2dcd3a5` in all eight
> drivers and `capstone-bootstrap` carries the fix.
> **`tests/monitor/scan-integer-bases.py` was run on BOTH sides of the bake** — one integer-derived
> base and exit 1 before, zero and exit 0 after — which is what makes "the fix is in the firmware I
> am about to boot" a different and stronger claim than "the fix is in the source".
>
> **The reclaimer does what it was built to do:** 200,000 allocations against a 65,532-index pool
> **without exhausting it**, and both cost curves flat — the ~12× release growth P1 measured and the
> capacity knee are both gone. Bundle at `experiments/results/M1/2026-09-17-reclaimer-pilot`,
> labelled a **platform pilot and not an M1 arm**, because M1's start gate is still one of three.
>
> **The console reports `caplifive_m1_054cea69b.bit`**, taken from a fresh read of
> `flash_state.nv_bitstream_name` rather than from the filename flashed, and confirmed to be a genuine
> gate match rather than the `None`-tolerating branch (zero `BITSTREAM IDENTITY UNVERIFIED` lines in
> the boot log — worth checking, because the drivers set that tolerance and a gate passing on it would
> have proved nothing about the string). All four places that named the old resident are updated:
> `known-good-controls.md`, `bitstream-usability-is-the-census-not-the-slack.md`, the launch doc, and
> the drivers' `FPGA_BITSTREAM` default.
>
> **A REFLASH INVALIDATES TWO CLASSES, AND ONLY ONE OF THEM IS GREPPABLE.** The first is every "the
> resident bitstream is X" line, which a search for the old name finds. The second is every statement
> about **what a board result MEANS**, and those live in documents that need not contain the word
> bitstream anywhere — the capacity bundle published that morning measures a knee that does not exist
> on the silicon now resident. It was caught only because the pilot happened to contradict it, and it
> was annotated rather than left standing. When a reflash lands, re-read the RESULTS of the last
> bitstream, not just the lines that name it.

## 2026-09-17 — CURRENT

> **A bitstream queued for flashing carries two fixes its name does not mention.**
> `caplifive_m1_054cea69b.bit` is the splice-plus-reclaimer build and **also contains R-34 and R-24**.
> Established by CONTENT: `misaligned_ex_q` in `cva6_mmu.sv` goes 0 → 5 against the flashed
> `1bfff7776`, and `DEBUG_REQUEST` goes 24 → 32. That makes capability exception delivery LIVE, and
> the monitor the drivers bake still computes its trap-handler writeback with an integer add on the
> stack capability — a dropped exception today, a delivered one inside the handler afterwards, on the
> `rdtime` path. **The D3 fix (`capstone-sbi` `d3-monitor-capability-writeback` at `2dcd3a5`) must be
> merged and the `4274268` pin moved in eight drivers before the first boot on the new silicon.** The
> memory map does NOT move — both capability-region constants are byte-identical at the two revisions,
> checked independently by two lanes — so the device tree stays valid.
>
> **The re-flash procedure is written down** for the first time
> (`docs/ref/HOW-TO-LAUNCH-ON-FPGA.md`, §RE-FLASHING); it had existed only inside the S-12 repro
> folder. Two corrections could not be applied in the list they correct, because
> `precommit-scan.sh`'s credential pattern matches a phrase committed in that list and blocks any
> diff that touches those lines, as context or as a removal. Both directions demonstrated. The file
> is frozen around that line and the fix is the lead's, since the pattern guards the real credential.

## 2026-09-16 — CURRENT

> **A RETRACTION leads this block.** The cold-regime revoke slope of **87.1 cycles per node is
> WITHDRAWN**: N = 2,048 sits only 1,239 cycles above the warm fit, so the 2,048 → 3,072 segment
> crosses the warm/cold knee and its slope measures the knee rather than either regime. The cold
> figure to cite is the board lane's **silicon** measurement, and it is a **range: 25–27 cycles per
> node walked** — Q4's capacity boot fits 25.45 (R² = 0.999969, eleven points wholly inside the cold
> regime) and the `M1_LIVE` sweep gives 26.96 over a different allocation range. **Both are silicon,
> both on the same machine, and they are not averaged**: two measurements taken over different ranges
> do not combine into one figure, and quoting either alone drops the other's existence. The 3.000 warm
> slope is unaffected.
>
> **AND AN INSTRUMENT RELABEL EVERY LANE NEEDS BEFORE ITS NEXT SIMULATION.** `S12_MEM_DELAY` is **not a
> cycle count**: `stream_delay.sv` has a 4-bit counter, so the value is truncated to its low nibble and
> the knob is a **period-16 sawtooth, not a dial**. The `40` used in all 39 places it appears — and
> described everywhere as "a 40-cycle memory" — realises as **8**. Measured totals on one tree and
> test: define 0 → 708, 2 → 1,415, 12 → 3,427, **16 → 1,004**. Turning it *up* from 12 to 16 turns
> latency *down* to near bypass, so anything ≡ 0 mod 16 asks for a large delay and gets essentially
> none, which reads as a clean negative rather than an obvious failure. **Usable range is 2..15.**
> This is a magnitude label, **not a retraction** — S-12, R-26 and R-34 all stand, dated to the
> revisions they ran at, on an 8-cycle memory. Read the define back from
> `work-ver/Variane_testharness__verFiles.dat`, never from the log.
> **2026-09-14 and 2026-09-15 are NOT in this file**; that block of board work is in
> `current-next-step.md`, whose 09-15 header stands.

* **R-12 is FIXED-IN-SIM on `m1-reclaimer` (`054cea69b`): the reclaimer is built and its approval test
  passes as a pair.** This is R-12's CAPACITY half — the 65,532-node ceiling — and it sits on top of the
  cost half below. Revoked nodes go on a LIFO free list folded into the write that already happens (a
  two-node REVOKE costs 252 cycles before and after); an allocation pops `(generation+1, index)` first,
  **+6 cycles** over a bump, and the allocator's call boundary costs **+1 per allocation** (confirmed at
  N = 65,532: the exhaustion fixture runs +65,542 cycles). Every use of a revocation reference now
  requires `valid && generation == g`. **The approval test is the pair:** on the gen-blind control
  `1ac15c4ef` a retained stale reference destroys the slot's new owner; on `054cea69b`, same tree and
  same fixture, every stale arm is refused with cause 25 and every fresh owner is intact — 8 traps
  against 4. An index retires after 16,384 reclaims rather than wrapping (measured at production width).
  An audit of the finished A5 diff found a composed id reaching a node's LINK, which let a generation
  SKIP past the retirement compare and wrap; fixed at both ends in `054cea69b` and measured as a pair.
  Sweep 65/27/3 with **zero status changes**; lint 733 with **UNOPTFLAT 40 unmoved**. **SYNTHESISED AND FLASHED 2026-09-17** — `054cea69b`, established by a LABEL-INDEPENDENT argument after an initial dispute: a run minting 200,031 nodes from a 65,532-index pool (same on both candidates) and reaching `stop=target` cannot have avoided exhaustion, which is a wedge pre-Part-B and a cause-30 trap after, so the image reclaims and the deployed one does not. **The console exposes no digest of the resident image**, so the cite-by-hash rule is unsatisfiable there; the workable form is hash-before-upload plus label plus a behavioural discriminator in the run. **Unresolved pending the image HASH**, which is the discriminator this project's own rule names; a label is not one. Any silicon measurement attributed to the reclaimer is provisional until then. S1 on `054cea69b`: **WNS −8.307** (best ever on this
  design), **combinational loops 1**, routed Total LUTs **168,757 — 456 BELOW the flashed base** while
  adding the allocator, the free list and the generation check; `capstone_rev_node` 985 LUT / 740 FF.
  All three pre-registered readings met. **Attribution withheld on purpose:** the branch is 14 `core/`
  files from the splice — reclaimer AND the r34-r24 merge — so no reclaimer-specific timing, loop or
  area claim is supported; `f714d2a72` is the only build that would split them and has never been run. The reclaimer is now the design on the board, and R-34/R-24 exception delivery went resident with it through the merge — capability exception delivery is live on silicon for the first time. Box: `ISSUES.md` R-12; note
  `docs/history/16-09-2026_20-30-00_m1-reclaimer-built.md`.
* **R-12's COST half is built, measured and synthesised; the capacity half above now rests on it.** The
  revoke-walk splice (`r12-splice-revoked-nodes`, submodule `f1331daed`, parent `9a5716d5ab6c`)
  unlinks a whole revoked run in two writes at the walk exit, independent of run length. Unspliced
  costs exactly **3.000 cycles per dead node crossed**, spliced exactly **0.000**: 399 / 516 / 481
  cycles at N = 4,096 / 6,144 / 8,192 against 167,498 / 294,367 / 403,353, a **839× ratio at the
  largest rung**. The attribution needs no model — the revoke that *kills* the nodes differs between
  the two trees by exactly +4 cycles at every rung, the revoke that *walks the corpses* by over
  400,000. Synthesis exit 0, WNS **−9.225** against the flashed −12.425, combinational loops 29 → 13.
  **NOT FLASHED**, and the 65,532-node ceiling is not addressed by any of it. Box: `ISSUES.md` R-12.
* **Two warnings travel with that synthesis result, and S2 has now split them apart (2026-09-17).** The
  timing gain is **not attributable to unlinking** — a runtime property cannot move a static loop count;
  the splice put the response send in two branches, so anvil registered the endpoint. **S2
  (`54ac25f97`, the semantically null duplication of that send on the unspliced tree) measured which
  half does what: WNS −8.684, better than the splice's and the best ever on this design, with
  combinational loops STILL AT 29.** So the endpoint registration buys the timing and none of the
  loops; the 29 → 13 drop belongs to the splice commit's structural changes. The earlier sentence
  "nine one-bit registers removed sixteen loops and bought 3.2 ns" is **retracted in its middle
  clause**. The ~642 LUTs outside the unit are likewise not intrinsic to the registration — S2 routes
  307 LUTs BELOW the splice — and remain unexplained. Quote implemented figures against implemented
  totals, never the +133 declared bits.
* **`reports/ariane.utilization.rpt` is overwritten after routing**, so in any archive of a build that
  routed it holds a POST-ROUTE number, not the post-synth one. Measured offset (S2): 171,620 post-synth
  against 169,637 post-route, −1,983 LUTs. This invalidated a post-synth LUT ceiling this lane had
  proposed for S1 — it would have killed S2, which routed fine — and the premise does not survive the
  correction either: the build that FAILED to route projects 163 LUTs BELOW the highest that routed, so
  **LUT count does not discriminate routability on this design**.
* **The sub-8 anomaly is the write buffer, measured rather than inferred.** Exactly two rungs sit
  exactly +91 cycles above the fit, at N = 6 and 7; halving the write-through dcache write buffer from
  8 to 4 moved the pair to N = 2 and 3 — a shift of exactly four — while every rung from N = 8 up
  stayed byte-identical. **The fit `3N + 249` is a bound in NEITHER direction below its range**: it
  over-estimates by 196 / 151 / 60 at N = 2 / 3 / 4 and under-estimates by 91 at N = 6 and 7. A small
  revoke has to be measured, and the answer tracks the buffer depth.
* **The reclamation v2 specification is committed, audited, and does not ship as written**:
  `docs/plans/2026-09-16-revnode-reclamation-v2.md`. The audit found three of its mechanisms
  individually wrong after the author had reviewed it twice. Item 6 — the three sites that grant
  authority without ever consulting a node — is still the fatal one, and Part B (graceful exhaustion)
  is separable and approvable alone.
* **M1's start gate is 1 of 3.** The lead named the RTL lane as runtime/RTL owner on 2026-09-16.
  Approval of the algorithm and of the stale-reference invariant are the remaining two, and cannot be
  given by the owner or the gate is decorative. **Nothing is scheduled and no reclaiming arm may be
  planned until both are given.**
* **The board is `apollo-board`'s; this session is backup, hands-off.** Their P1 no-reclamation
  baseline ran four boots, all `done`, and refuted its pre-registered primary in the informative
  direction: scored apart, `take` is near-flat at ~70 cycles while `give` grows 221 → 1,558 and
  superlinearly in cumulative allocations, against flat-at-16 with slope exactly 0.00 under the
  emulator. Queue, gate order and the lead's open decisions:
  `docs/plans/2026-09-15-consolidated-board-queue.md`.
* **D3 IS CLOSED: the monitor's one plain access through an integer base is fixed and validated.**
  `capstone-sbi` `d3-monitor-capability-writeback` at `2dcd3a5` moves the stack capability's cursor in
  place rather than computing an integer address from it. Validated where it CAN be — against the
  R-34/R-24 delivery-fix branch, because on the deployed bitstream the old form and the new one both run
  clean. Matched pair at `c77c65324`, 601 cycles: the replacement takes cause 0 and its store reads back,
  the capability is bit-identical before and after, the old shape run last takes **cause 24** with its
  store refused, and there is **exactly one trap in the run — the control's**. The replacement is bounds-checked where the integer base was not, so a lower-edge arm tests the one address the preceding `SAVE_REG`s do not (slot 0, `rd` = `x0`): cause 0, value stored. The residual is runtime state — whether the live stack capability's base reaches `frame_base` — and only a boot on a delivering bitstream settles it. The branch is held OFF
  `capstone-bootstrap`, which stays at the commit the drivers pin, so a boot today bakes the monitor it
  expects, and is **now published** — the push allowlist was retired on the lead's instruction on
  2026-09-16, so a task branch no longer waits on a file edit. The hook still blocks shared history,
  deletion, non-fast-forward and `capstone/paper`, which are the four that were ever dangerous. Note found on the way: `RVTEST_PASS` stores to `tohost` through an `auipc` integer base, which
  is the same defect and why ten of the twelve sweep tests time out in their epilogue; exiting through a
  capability minted over `tohost` works. Instrument, readings and note:
  `tests/monitor/`, `docs/history/16-09-2026_15-00-00_d3-monitor-writeback-validated.md`.
* **The stale-artifact trap now fires mechanically** (`a0d83f6b8a14`). A worktree generates its anvil
  output at creation time, so a source patch applied afterwards leaves every local gate testing the
  pre-edit design: a 95-test sweep read as "the change is inert" when it meant "the change is absent",
  and the lint numbers and two cost measurements taken beside it were void the same way.
* **A gate in `board-c6var.sh` could not fail on the variable it existed to pin** (fixed 2026-09-16).
  It demanded an emulator record by its `HEAP` field, which is `sublet_tables_len`, computed from the
  GRANTED ARENA alone — so it witnessed the arena and never the `--tables` grant the boot also sets.
  A record from the committed flow at the 2 MiB arena carries tables 2,523,136 against the boot's
  1,750,285 and prints the identical `HEAP 1344064`: indistinguishable by construction, a clean pass
  against a denominator from another configuration. Arena and tables are now one variable feeding both
  the gate and the boot, and the gate also demands the measure flow's configuration line, which names
  both. Negative-tested: the old gate passed the wrong-tables record, the new one refuses it. Found by
  the compiler lane, verified on apollo, driver half fixed here.
* **Four result bundles on the paper remote were reviewed rather than accepted** (`7e77374aa2b5`),
  including one that keeps all seven entries while recording the loss of its own raw evidence. The
  packaging hole found there generalises to bundles this lane cannot see.

## 2026-09-13 (evening) — EARLIER

> sw64's stall reproduces on an exact redraw and is in the domain's share entry; the mtvec pair
> is CONFIRMED end to end (sw68 fix + sw69 mcause-27 readback); ten collaborator PRs landed (all but capstone-qemu #3). The morning block below
> stands for the bridge and R-30/R-31/R-33.

* **sw66 reproduced sw64's stall exactly** — `F2/share3 → SHA5`, no `SHA6`, no `G/enter` — on the
  manifest-verified artifacts, second draw. Deterministic for this image; per the monitor's own
  marker definitions the hang is inside the domain's share-entry execution with `mtvec = 0`. Its
  baseline arm returned the **size-20 denominator: 54,230,566,323 cycles, hash `3807866 2738af78`**.
* **sw65 said nothing about sw64** (three variables changed; its image had never run on QEMU —
  the runner's QEMU phase had no `cma=` and died before entry, and the image was staged anyway; it
  runs clean there today at `cma=256M`). **The S-15 double delin is not the discriminator**: same
  code in both images, and sw65's passed share3 with it.
* **The pair is proven, not asserted:** `cc55013c2106` from the worktree reflog; the no-flag rebuild
  hit `23da3b126a304585` / `7c27697818b0abe0`; the mtvec image `214b300efd169f03` has its QEMU
  licence. Banked at `~/capstone-artifacts/sw64-pair/`. **sw67 ran it: share3 RETURNS (`SHA6`)
  where sw66 hangs, then `sqlite3_initialize` fails (`0x5117BAD3`, = sw65). S-15's account
  strengthened** — audited as not yet a measured root cause (share3 also differs in region residency
  and size), but its mechanism is now sourced at the RTL commit (DELIN raises on any non-LINEAR
  operand; the only type-sensitive instruction in the branch). The trap word went into the arena
  and sw69 READ IT BACK: `arena0=0xF6C09D13`, mcause field 27 (§7o). Fix on `dev`,
  **proven on silicon by sw68**: the image with the delin removed and nothing else, no trap vector,
  passes share3 and enters where sw64/sw66 hang, and ran size-20 to completion: domain
  **64,732,455,367 cycles**, hash `3807866 2738af78`, **ratio 1.1940** on the 54,214,856,567 baseline
  — inside the pre-registered 1.17–1.27 band. The fix runs the real benchmark end to end on silicon.
* **Boot sw73 (2026-09-14): `main --size 100` on silicon, the default size, both arms complete at the
  native oracle `23674002 573a4409`: native 337,235,381,252 cycles (3.75 h), domain 398,572,346,349
  (4.43 h), **ratio 1.1819 → 1.18**, exactly the pre-registered value (band 1.14–1.24); instruction
  ratio 1.2383 × CPI ratio 0.954. Size series on silicon: 1.2195 / 1.1940 / 1.1819 at sizes 1/20/100.
  Every invalidator checked before the ratio was written; §7p carries the caveats (timing, lookaside off,
  cycles only).
* **2026-09-14, the plan's four boots (§7q):** the Sublet cell re-based on the #3 module (2,794,183,730
  cycles at HEAP 911104; silicon ⑥/⑤ 1.0951), the lookaside-ON arms on silicon (baseline 2,108,202,651 at
  size 1 with 25,122 lookasides), the rounded-arena reclaim completing through csinit (audited: R-33's
  account, attribution inferred), the entry watchdog firing live on sw64's stall, and **`main --size 20`
  with lookaside ON on both arms at 1.1937 → 1.19** (the OFF pair: 1.1940) — SQLite as it ships costs the
  same on this silicon. Two per-boot rules learned the hard way: one Sublet workload per boot (R-12,
  43k rev-node mints per run) and one REGION_ARENA workload per boot (a second 128 MiB arena creation
  stalls the host; mechanism open).
* **Watchdog fixed** (`f7f2c9030623`): liveness is `[uart]` lines; console `[event]` chatter had
  made the entry-stall abort unfireable live.
* **PRs landed on `dev`: ten of the eleven original, and all seventeen new ones (#19–#35).**
  Original: #15, #16 (+print fix), #11/#12/#13, buildroot #2/#3 (submodule `capstone-bootstrap`=`d04bd83`,
  module rebuilt + 3 QEMU controls: the ioctl-struct grew so every board host must be rebuilt with it),
  #17 (+follow-up), #18 (gate PASS=4), #14 (`d616ea4e` + follow-up: the rebuilt 160-test image differs
  by exactly 17 instructions, all `sd ra`→`stc ra`/`ld ra`→`ldc ra`; claim-auditor SUPPORTED; lit 93/93,
  authority 32/32, sqlite-silicon + MicroPython PASS on the rebuilt toolchain). **FPGA-side #3 LANDED,
  proven by boot sw71** (control `retval=4` with the rebuilt controller — its private ioctl struct
  grown, f308efe2 — and the declaring SQLite image running to the size-1 oracle on the new module;
  sw70 before it was VOID: the native baseline controller had been staged into the `lpc` slot). New,
  2026-09-13/14: standalone #20/#21/#28; the nginx stack #29–#35 in order (pool gate 90/0; the
  use-after-destroy pair reproduced, fault at pc 0x166cc; #35's replay not reproducible here — no trace
  file locally, recorded as the collaborator's claim); the MicroPython stack #24→#25→#26→#19→#22→#23→#27
  as plain merges of the collaborator's rebase (0 replayed commits; gates on the tip: default PASS=4;
  full level 557 rows PASS=551 FAIL=1 FAULT=0 SKIP=5; weakref-on 563 rows PASS=552 FAIL=1 FAULT=5
  SKIP=5 — and the gate's `MPY_GATE_TESTS=all` mode cannot fail, its judge compares to the literal
  `all`; recorded in the merge message and the hand-off note). The eight other freestanding hosts with
  the private ioctl struct are grown (B2, 2026-09-14). **Held:** capstone-qemu #3 (rebase); #14/#18's
  force-pushed rebases are to be CLOSED, not re-merged (landed by content).
* **The toolchain binary is STALE** per `toolchain-fresh` (now that #15 lets it say so): the
  `opt`/`llvm-symbolizer` targets were never built. Rebuild at #14's step, never during a suite.

## 2026-09-13 (morning) — EARLIER

> The bridge is discharged. §7f–§7k now carry forward to the flashed bitstream; the 2026-09-12 block
> below stands for R-30/R-31/R-33.

* **THE BRIDGE HOLDS (boot sw63).** §7k's images re-run **unchanged** on
  `caplifive_r30r31_1bfff7776` — verified byte-identical after the bake, from both `overlay/` and
  `build/target/`, which mattered because the overlay was holding the lookaside matrix builds under
  those very names. Every §7k-comparable pair agrees to within **0.07 pp** against the 0.171 pp
  cross-boot band, and `main`, the pair the protocol names, to **0.05 pp** (1.2195 vs 1.220).
  Controls at both ends, **7/7 pairs agree on their verification hash**, `DROPPED 0` throughout.
  **§7f–§7k are carried forward.** Details in §7m. This is the arm sw59 was wrongly reported as.
* **R-33's fix was inert on that boot, and it was proven rather than assumed** — zero
  `not representable` lines. Every constant on the path is a power of two. It would **not** have
  been inert for a `--pool`-derived arena, where both the arena and tables round.
* **The size-20/100 artifacts were rescued from `/tmp`** into
  `~/capstone-artifacts/speedtest1-size100/` with a `SHA256SUMS` and a README. They were
  single-copy, including the only record of their hashes. One set covers sizes 1, 20 and 100 — the
  size is argv and the arena lives only in `HOST_EXTRA_DEFS`, so no rebuild is needed for depth.
* **The `speedtest1` branch is merged into `dev`** (`5e5d9cb42318`). It was on no remote and not on
  the push allowlist. That lands §7l **and `arena-mismatch-gate.py` with its runner wiring** — until
  now a campaign from the main checkout ran with no arena gate at all, and the mismatch it catches
  silently corrupts a ratio instead of crashing a run.

## 2026-09-12 — superseded above for the bridge; current for R-30/R-31/R-33
## 2026-09-18 — CPython allocator component

`capstone/ports/cpython/pymalloc/` adds CPython 3.13.7's actual pymalloc to the
shared port template. Native reference comparison covers normalized allocation
decisions and payloads. A 123,622-event native JSON/regex/bytearray recording
completed in both spatial and Sublet QEMU replays; the 18 paired lifetime cases
passed, with fault verdicts checked at the intended access instruction.
This is allocator replay, not interpreter execution or FPGA measurement.
The component README describes the extraction boundary, fixed backing budgets,
raw fallback, recorder limits and validation commands.

## 2026-09-12 — CURRENT

> Three boots on the R-30/R-31 bitstream. The 2026-09-10 block below is still accurate for the
> firmware half; everything it says about R-30/R-31 being unverified on silicon is now superseded
> by this block. Measurements: `ref/fpga-silicon-measurements-for-paper.md` §4g.1–§4g.5.

* **The bitstream is flashed and verified BY CONTENT.** `caplifive_r30r31_1bfff7776.bit`,
  `nv_bitstream_sha256 = 406e12bf…3b30` read back from a fresh `/api/state` after the mandatory
  power-cycle. Control `k800` = 4 in **all three** boots (sw59/sw60/sw61), `instret` 1089 in every
  one, cycles 4521/4517/4517 — so the flash did not move timing. `known-good-controls.md` is
  refreshed against it; only the `k800` row, the others still carry older bitstreams.
* **R-31 is FIXED ON SILICON** (boot sw60), through the monitor's real share/revoke path rather
  than a fabricated capability: `SHA2:00000003` = cap_type UNINIT where the previous bitstream
  returned LINEAR, and `RCPR` did not fire, so the cursor is at base too. Both halves of the
  contract hold.
* **R-30's headline is SUPERSEDED, and this was over-stated once before being narrowed.** The
  one-byte precondition is fixed — boot sw61 performs **5,334 successful INITs**, every counter
  bit-identical to QEMU. sw60's `RCSH:000006C0` is a *separate* large-region effect one step
  earlier, where INIT refusing is correct. Two of the four candidate accounts are dead: the
  monitor's own arithmetic admits a shortfall of at most 15 bytes (which also kills
  allocator-rounding), and the RTL kills "end moved during the fill". **One account survives: 108
  stores did not advance the cursor.** The discriminator is a pair of boots at two different large
  region sizes — constant 1,728 = a fixed tail effect, scaling = a proportional store-failure rate.
  Not yet run.
* **The silicon allocator matrix is complete.** ABI cost **~1.21**, with the two allocators
  **indistinguishable at this precision** (④/① 1.2124, ⑤/② 1.2107 — 0.17 pp against a 0.171 pp
  cross-boot band, so not resolvable either way; the deterministic QEMU pair *is* resolvable and
  shows lookaside costing marginally less, so "barely depends", not "independent"). The Sublet
  **CONFIGURATION** costs **1.0964 on silicon against 1.0176 on QEMU** — 5.5×, consistent with an
  O(bytes) reclaim. **Not "the discipline":** both pairs carry the heap-geometry mismatch for which
  the QEMU figure was already retracted as a discipline cost, so the 5.5× is suggestive of the
  mechanism rather than a measurement of it. (Both corrected 2026-09-12 after a bench-lane audit.) Two caveats travel with these rows: the ⑥/⑤ comparison
  carries a heap-geometry term (910,008 vs 2,097,152, not equalisable), and the lookaside-ON rows
  must not be blended with the lookaside-OFF §7 corpus.
* **R-33 is a SOUNDNESS issue, ISA-LEVEL, and its cause is the ALLOCATOR.** Demonstrated 2026-09-12 to reach ordinary LINEAR capabilities through `CINCOFFSET` — plain pointer arithmetic — by a matched RTL-sim pair where the representable control does not move and the non-representable arm widens by exactly the predicted amount. The
  rounded `end` is the authority bound — `STC` checks `rs1_up > metadata.end - 16` against the
  decompressed value — so a non-representable region grants writes past itself, and `CINCOFFSET`
  reaches ordinary linear capabilities by the same route. A lossy compressed-bounds format is
  standard and is exact for representable objects; nothing here enforces representability, and
  `create_region(N)` passes `N` through unchanged. **Contained by the kernel's `PAGE_ALIGN` below
  4 MiB and not contained at or above it** — no region used so far escapes (sw60's sits exactly on
  the boundary), but a 130 MiB-class region has a 262,144-byte granule. The over-permissive store is
  **DEMONSTRATED in RTL simulation 2026-09-12** — a representable control's store at its true end is refused OUT_OF_BOUNDS while a non-representable arm's identical store retires without fault (`r33-store-past-end.S`, trap_mask 0x1 as pre-registered). Not yet shown **on silicon**, and the bottom-truncation half is still unexercised.

* **R-30's residual is SOLVED and re-filed as R-33 (boot sw62).** The 1,728 bytes were never a
  failed fill. A capability's bounds are re-encoded once its cursor leaves `base`, and `end` then
  reads high by up to one granule — `compress_bounds` uses an exact form only while the cursor sits
  at the low bound (`ariane_pkg.sv:787`) and otherwise rounds the top up to `2^(E+3)`
  (`:827-828`); `STC` is a DYN op (`decoder.sv:1309`) whose `rs1` is re-compressed on writeback
  (`ex_stage.sv:1188`). sw62's granule-ALIGNED arenas reclaimed clean, while the unaligned arm
  halted with `RCSH = 448` — the compression figure, pre-registered before the boot against 432 for
  a store-failure rate — with `RCCU` showing the cursor reached the true end, so **no store failed**.
  Two accounts were retracted on the way there, both this lane's and the RTL lane's, and both had
  read the fat struct without the function that compresses it.

* **The resident firmware is THREE monitor commits behind and none of them has booted** —
  `75d96d2` (define `CAP_TYPE_UNINIT`), `921f598` (`RCEN`/`RCCU` reclaim instrument), `d1bd7e4`
  (early-clobber on both `C_RECLAIM` outputs). All boots ran `2c49c41`. `board-b59/b60/b61.sh`
  gate on that hash and will now FAIL — correctly; update it deliberately, do not delete the gate.
  The submodule pointer chain is deliberately unbumped, which matters only for a fresh clone.
* **`precommit-scan.sh` had a silent-pass path** — a `--range` git could not resolve contributed
  nothing and printed CLEAN. Fixed (unresolvable *or* empty range now blocks), negative-tested four
  ways. Nothing had escaped it.

## 2026-09-10 — superseded above for R-30/R-31; **2026-09-11 is NOT in this file**

> **Read `state/current-next-step.md` first for 2026-09-11.** Four boots landed that day and none of
> them is described here: sw55 (a **130 MiB** capability region on silicon), sw56/sw57/sw58
> (speedtest1 across seven testsets, the domain's instruction count measured on silicon, and the
> position question settled). The measurements are `ref/fpga-silicon-measurements-for-paper.md`
> §7f–§7k. Also that day: the CMA board half, M-6 fixed and M-7 filed, the R-30/R-31 **firmware
> half committed at last** (all four monitor pointer paths had been committing its pre-reclaim
> parent), and the discovery that **six repositories refuse this credential** — including both
> copies of the monitor and the academic spec, which returns 403 on read as well as write.


* **The reclaim (R-30/R-31 firmware half) is implemented, gated and measured.** The lead ruled
  *fill, then initialise*. Monitor commit `0a5c3d9` adds a `C_RECLAIM` asm loop at **five** sites
  (the annotation branches share a hoisted one, so REV_DEFAULT/BORROWED/SHARED/TRANSFERRED are all
  covered), with the loop bound taken from `cap_end - cap_base` and **not** the cursor, so a
  cursor-at-end arrival faults instead of running zero iterations. Emulator gates green: host-call
  12/12, linear/uninit corpus 7/7, smoke, nullblk 3/3. ~~**`0a5c3d9` is LOCAL ONLY**~~ **CORRECTED
  2026-09-11: it is PUSHED.** `git ls-remote` — the remote itself rather than a cached
  remote-tracking ref — shows `capstone-sbi refs/heads/capstone-bootstrap` at exactly
  `0a5c3d9a3413`. The 403 recorded on 2026-09-10 was real then and was carried forward for a day
  without being re-tried; nobody needs to push this.
* **Boot sw52 — speedtest1 on capability silicon vs native, plus the fill-cost pair.** Control
  `k800` = 4, zero fault tags, 10 of 11 arms. Capability/native **cycle** ratios 1.335 / 1.181 /
  1.214 for parsenumber / orm / main, with **identical verification hashes** on every pair; the cycle
  ratio is BELOW the instruction ratio in all three because capability CPI is *lower* than native's.
  Full numbers: `docs/ref/fpga-silicon-measurements-for-paper.md` §7f.
* **Boot sw53 -- the fill-cost pair with an error bar, and both of §7e's open questions closed.**
  9/9 arms, `k800` = 4, zero fault tags. **24.1 cycles per 16-byte capability store at n=3**
  (within-boot spread 0.2-0.5 %; sw52's single draw of the earlier rung gave 23.6). A 4 KiB reclaim
  is **+3.6 % instructions and 7.8-9.3 % of cycles** at speedtest1's measured CPI, 4.6-26.2 % across
  the full 1.13-6.44 spread -- CPI-sensitive, not "in the lower half of the bracket". An adversarial
  audit had already moved that figure from 5.5-6.6 %: the first version counted the fill loop's
  INSTRUCTIONS but only the stores' CYCLES, and the monitor pays for the loop too.
  - **A warm region is NOT materially cheaper** (`fillwarm`: second pass >= 87 % of the first), so
    the cold-buffer "this is a ceiling" caveat is deleted and the figure applies to the monitor's
    real case.
  - **The cost is per STORE and is ~96 % not capability-specific** (`fillsd`: a plain 8-byte store on
    the same 16-byte walk costs 23.1 against `stc`'s 24.1, writing half the bytes). It is the
    write-through drain, not the tag.
  - **R-3 does not bite the ladder path**: every rung repeated at its own entry VA returned.
  Full numbers: `docs/ref/fpga-silicon-measurements-for-paper.md` §7e, §7f.
* **`RCLM:00000000` on every share (9 of 9).** Zero reclaims, as it must be where the guard cannot
  fire — and the counter's reporting path is now proven readable on silicon, which is the one
  property of it that could not be tested after the flash.
* ~~**Still open and the lead's:** the `end`-convention re-ruling.~~ **RULED 2026-09-10 (evening) —
  adopt the resolution**; `plans/DECISIONS-WAITING-2026-09-10.md:50`. This line said "still open" for
  a day after the ruling landed, which is the same shape as the stale 403 withdrawn on 2026-09-11: a
  blocker asserts a fact about today, so re-verify before repeating it. The spec amendment is
  committed in `capstone-academic-spec` (`21a01f0`, `eadaf87`) and **cannot be pushed** — that remote
  returns 403 for this credential on read as well as write, so its branches cannot even be listed.
* Three tools repaired and negative-tested the same day: `preflight-board-run.sh` (parsed only one of
  the three stage forms, and could not require a host binary at all), the sw52 result parser (needed
  the testset, cycles and hash on ONE line when they are on three), and the driver's staged-marker
  guard (did not know the `BASELINE-PROBE` form, and cost sw52 its last arm).

## 2026-09-09 — SUPERSEDED BY THE ABOVE, still accurate for what it covers

* **The R-25/26/27 bitstream is on the board** (`caplifive_r25r26r27_66c4e7517.bit`, flashed persistently
  2026-09-09; `caplifive_s12fix_5097eb166.bit` stays in the console store as the restore path). Boots sw44
  (closing set 9/9), sw45 (acceptance 7/7, R-25 PROBE wedges as predicted), sw46/sw47 (firmware variants D/E
  clean — the R-26 board arms, no-regression only), sw48 (the s06agg isolation). R-25/R-26/R-27 archived.
  The monitor drops its four R-26 `fence.i` (monitor `1a39e37`, wrapper `f17110a`, buildroot `fe31893`,
  caplifive-system `884b716`; nested pushes need the lead's credential).
* **RETRACTION: S-06's struct-assignment half was never fixed — now `R-29`.** `s06agg` reads 66 on the
  new bitstream in two firmwares (sw46, sw48); the RTL lane's directed test fails on `5097eb166`,
  `ef5a8eaf2` and `66c4e7517` alike when the plain `sd` of the high half is adjacent to the 128-bit `ldc`
  (write-buffer forwarding, S-07/S-10 family). The S-06 folder's cited acceptance (boot B1 "15") was a
  different program. Memcpy half stands. `W-12` stays. Not a regression of the new bitstream.
* Driver classifier: `created`/`entered` now also read the monitor's `DBAS`/`ENT1` tags, so a wedged
  `rtpc`/`lpc` probe is no longer reported as "domain never created" (sw45/sw47 r25dup).


* **The monitor stack is unified onto ONE branch, `capstone-bootstrap`, in every nested repo**
  (caplifive-buildroot `b7fc740`, opensbi `3de3342`, monitor/sbi.dom `3da7ebe`, caplifive-system
  `fec33fa`; parent `dev` `1a1a3dff5778`, all pushed). One source builds both targets:
  `make TARGET=fpga|qemu`, output in `build-<target>/`, `build` a per-checkout symlink,
  per-target code under `CAPSTONE_TARGET_FPGA`/`CAPSTONE_TARGET_QEMU`. `CAPSTONE_CC_PATH` is
  required. The old two-branch split (board vs QEMU, drifted for six weeks and forced Q-03/Q-05 to
  be ported twice) is gone; the `-board`/`-qemu`/`-unified`/`-dts-65536` names are frozen pre-merge
  tips (ancestors of `capstone-bootstrap`, also `pre-unify/2026-09-08/*` tags). Design and record:
  `docs/plans/monitor-unification.md`; layout: `docs/ref/REPO-MAP.md`.
* **Validated on both targets.** FPGA `.c.S` pair, `fw_jump` and `fw_payload .text` byte-identical
  to boot sw31; **board boot sw32 8/8** (control first, six BEEBS rungs, SLT `select1` identical to
  native, zero fault tags), with the SQLite host program rebuilt against the merged loader library
  so that library has now run on silicon; **QEMU nightly tier 18/18**. sbi.dom now builds from the
  one monitor source (Phase B item 7 done).
* **One nightly regression, surfaced and fixed the same day.** `linear-uninit-corpus` /
  `linear_drop_sibling_ok` failed; bisected to the Q-05 monitor commit (a pre-existing stale
  host-observer read the unification's first nightly exposed, NOT a merge defect), fixed in the
  corpus controller to read back through the domain's alias. Auditor-confirmed; a non-blocking
  monitor-robustness note recorded (ISSUES.md Q-05, 2026-09-08).
* **Phase B collapsed the per-target behaviour (2026-09-08, evening and night).** Geometry on QEMU
  (item 3), M-2 bounded at 96 (item 9, refusal seen on silicon), the pre-carve refusal on FPGA
  (item 1), the rounding and the diagnostic store (6a/6b), eleven of fifteen `fence.i` gone (item
  8, three variants booted), the null-blk package and the relocatable S-mode loader (item 4), ONE
  `create_domain` (item 5, nine differences gone) and the transferred slot a hole on the board too
  (item 2: boots sw36/sw37 ran the first transfer-annotated share on silicon, one HOLE line).
  Board boots sw33–sw37 all at the oracles, zero fault tags; QEMU tier 18/18 through item 3,
  17/18 on item 5A (one BEEBS case silent before the loader's first line amid five boot-login
  infra flakes; 3/3 rerun alone, first in a fresh boot). Left: kernel
  unification (item 10, deferred by the lead), the dead `mem_l`/`mem_r` locals, and the
  `gpoff == 0` loader branch that no board image reaches. Monitor 5b27d01 / buildroot d3c2402.
  **Closed on the shipping firmware 2026-09-09: boot sw38, 9/9** (control, six rungs, SLT select1,
  the transfer probe; one HOLE line; zero fault tags). Next: `docs/plans/after-phase-b.md`.

## 2026-09-05

* **SQLite passes its logic tests on silicon at `-O1`** — the first validation above `-O0`.
  `select1` 1031 records / 1000 queries / 0 failures and `q_two` (the S-12 trigger) both completed
  in a capability domain on `caplifive_s12fix_5097eb166.bit` with the cycle-2 compiler, valid
  control first. Full sweep: **8 boots, 8 valid controls, 19 rung readings, every rung at its
  oracle** (`tests/board-results/2026-09-05.tsv`, compiler lane's branch). RV8 `-O2`, CoreMark
  `-O2` with sibling calls, BEEBS `-O2`, two csmith rungs — all at oracle. C-28's tail-call fix runs
  on silicon, so `-fno-optimize-sibling-calls` can be retired.
* **S-12: 6 of 6 post-fix draws clean, p = 0.033** — see ISSUES.md; the two new draws are `-O1`
  and therefore weaker, so this is strong evidence and still not "proven".
* **The gp-captable miscompute (OPEN since 2026-07-23) does not reproduce** — `rc_p1` = 2080 at its
  oracle. Probable cause **R-20, fixed in hardware by `f623c48a1`**, whose signature is exactly
  that bug's. The blocks it carried (silicon-compatibility claim, branch merge, app-level silicon
  perf) are no longer supported by a live failure.
* **R-20 is FIXED in the resident bitstream** — an alert claiming otherwise was filed and
  **retracted** the same day: the fix is a cherry-pick under a different SHA, and both lanes had
  tested ancestry by hash. Presence-by-content is the check; see ISSUES.md.
* **S-13 does not reproduce at `-O1`**, but bitstream and compiler both changed, so it attributes to
  neither yet.
* **2026-09-07:** Q-03 ported to the BOARD firmware (`fw_payload 44c88d9ebeb1`, audited; boot sw30 7/7 in one
  boot, no exact fit occurred so the hole path is unexercised on silicon and self-reporting); Q-05 fixed in the
  stand-in (the probe observes through the domain; both copies make the transferred slot a hole).
* Q-02 (QEMU build break) closed end to end; Q-03 (position-dependent wedge, reproducible off-board),
  R-25 (INIT linearity break), C-41 (compiler `return` encoding), I-01..I-03 filed and verified.

## 2026-09-04 — superseded by the section above

* **Bitstream: `caplifive_s12fix_5097eb166.bit`** (sha256 `7a97ccd0…62999b0`) — the S-12 fix,
  synthesised and flashed 2026-09-04. It IMPROVED timing over its predecessor: WNS −16.400 →
  −15.311, 987 fewer failing endpoints. Every silicon number taken before it should name the
  bitstream it was taken on.
* **S-12: ROOT-CAUSED, FIXED IN RTL, FLASHED — "consistent with fixed", NOT proven.** A capability
  store's scoreboard rd is aliased to its own store-data register; when it stalls on a full store
  buffer the commit stage holds `we_gpr` while withholding `commit_ack`, the WAW guard clears on
  that write, and forwarding hands the consumer `create_cnull()`. The write happens; the
  RETIREMENT does not. Fix = require `commit_ack_i` in both WAW-clearing clauses, four lines.
  Post-fix the SQLite domain completes 4 draws of 4 against a pre-fix arm that trapped 3 of 4 —
  Fisher p = 0.071. **Two more draws would settle it; until then do not write "fixed" unqualified.**
  Full mechanism: `capstone/tests/fpga-repros/S12-wherecode-notcap-operand-vs-memory/S12-explanation.md`.
* **QEMU is REPAIRED and rebuilt (2026-09-04).** The c128 merge had left `capstone-qemu` unable to
  compile, and because nothing rebuilt it, every QEMU verdict for a day came from a binary dated
  2026-08-27. Three defects fixed in `f5972c364f`; smoke passes and the SLT negative control
  passes, so the comparator is proven able to fire. See `ref/ISSUES.md` Q-02.
  **Two things still do NOT follow from that fix.** The "SLT corpus matches native 15/15" figure
  has no committed harness — it was run ad hoc, so a rebuild does not re-establish it; treat it as
  withdrawn until a re-runnable harness exists. And the nightly still cannot catch a
  non-compiling QEMU. SILICON results were never affected: they came from the board.
* **SQLite CORRECTNESS on silicon: the SQLLogicTest corpus (2026-09-05 → 09-07).** Seven files,
  one boot each, control first, fresh toolchain, every result compared with the native baseline
  from the run's own transcript: negative control and `aggfunc` reproduce their known failures
  exactly (the comparator fires on silicon); `select1/2/3/4/5` identical to native with zero
  divergences — **the whole corpus, 10,807 records, 8,746 checked queries, on silicon.** Rows sw23–sw29 and B8 in
  `tests/board-results/2026-09-05.tsv`; write-up in `ref/fpga-silicon-measurements-for-paper.md` §7c.
* **(Superseded by the line above, kept for the caveat it carries.) SQLite RUNS ON SILICON — that is a LIVENESS result, not a correctness one.** The `slt/`
  corpus executes end-to-end in a capability domain; `s12stress` completes 120/120 prepares and
  15/15 of the corpus matches native under QEMU on the current compiler.
  **Read what that measures.** These files are S-12 *wedge probes*, and they say so in their own
  first lines — `p8_trivial.test`: *"WEDGE PROBE, not a correctness test: expected values are
  dummy, the signal is RETURNED vs WEDGED."* Every table in `s12stress` is deliberately EMPTY,
  because S-12 fires at PREPARE time with no rows processed. So "matches native" is a strong claim
  about **completing without wedging** and a nearly vacuous one about **computing the right
  answer** — the queries mostly return nothing on both sides. Establishing SQLite *correctness* on
  silicon would need a different corpus with populated tables and real expected values, and that
  has not been run. Do not let this line become the citation for a correctness claim.
* **C-19: RESOLVED.** Reading a capability's address now uses a plain move, never `lcc rd, rs, 2`,
  which is not total and traps on an untagged (NULL) operand.
* **The c128 capability value type is MERGED** (external collaborator's branch, 2026-09-04).
  `MVT::c128` replaces i128 as the carrier. Merging it silently reverted C-19 and three header
  declarations; all repaired — see the merge commit. One known coverage gap remains in
  `ptr-diff-signed.ll`.
* **S-06, S-07, S-08: fixed and verified on silicon** (see the 2026-08-16 section below).
* **The debug instrumentation is STALE and expensive.** Every mux reading across the S-12 campaign
  was weak, void or faulted — its own decoder says "UNKNOWN SEMANTICS for this bitstream" — while
  costing 1.820 ns, more than the S-12 fix gained. Every verdict came from software instead.
  `plans/instrumentation-cleanup.md` is now unblocked.

**Next steps are in `state/current-next-step.md` §0.** Sections below this one are retained as the
historical trail; the newest of them is dated 2026-08-16 and predates all of the above.

---

---

## Everything older

The historical trail — the append-only layers from 2026-08-16 back to June, including the S-06 /
S-07 / S-08 bring-up, the R-18/R-19 handovers, the UART retirement and the original overhead
tables — is preserved verbatim in
**`history/04-09-2026_17-00-00_current-state-historical-trail.md`**.

It was split out on 2026-09-04 because this file is the first thing every session reads, and 97%
of it described states that two RTL fixes and a reflash had already invalidated. Nothing was
deleted. When this file and `ref/ISSUES.md` disagree about a defect, **ISSUES.md wins** — it is
the registry; this is a snapshot.
