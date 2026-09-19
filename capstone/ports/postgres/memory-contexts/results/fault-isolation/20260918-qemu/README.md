# QEMU client fault isolation, 2026-09-18

Opt-in cooperative recovery, with the runtime and tests in this commit and
QEMU `77d69353b7`. [inputs.json](inputs.json) fingerprints the tested binaries,
compiler, emulator, inputs, region geometry and verdicts. The emulator was built
from the same source before its commit; the recorded binary hash identifies it.
Raw serial captures remain local, not committed.

| Check | Result |
|---|---|
| Native CTest suite | 14/14 pass, including negative verdict controls |
| AllocSet, Generation, Slab, Bump healthy clients | 8/8 spatial/Sublet arms pass |
| Bounds, tag, bad SP/GP, region-share fault | 4/4 exact-site fault returns |
| Three calls per quarantined domain | Client entered once; post-fault access never executed |
| RWX-vector negative control | VM fail-stop retained; no local delivery |
| Broken recovery-context negative control | One delivery, then VM fail-stop; no recursion |
| Same-boot process isolation | Healthy → three SIGSEGV children → healthy |
| Paired reset/bounds security checks | 12/12 arms pass; 6 exact-PC delivered faults |

[context-checks.json](context-checks.json) records the additional paired checks
and their binary/input hashes. In these standalone expected-fault arms the
generic runner returns 1 because the normal completion marker is absent;
the classifier requires the launcher fault marker and guest exit status 139.
The separate same-boot supervisor above establishes actual signal termination.

The three children are Generation, Slab and Bump clients dereferencing an alias
after context reset. Each fault matched its declared access PC. A separate Linux
supervisor checked `WIFSIGNALED` and `WTERMSIG == SIGSEGV`; an ordinary exit code
139 is not accepted. The final healthy Generation client completed after all
three deaths, in the same VM boot.

The fallback arms intentionally stop QEMU: their runner status 1 is expected,
and only their exact fault signatures make them pass. They are not process-only
recovery claims. The recovery suite passed on repeated runs, including the final
configuration with an additional 256 bytes declared for recovery state.

Limits: no firmware/kernel fault containment, FPGA claim, adversarial CSR-tamper
resilience, general domain destruction, or complete resource reclamation. Recovery
is disabled by default; see [the runtime contract](../../../../../../runtime/domain-faults.md).
