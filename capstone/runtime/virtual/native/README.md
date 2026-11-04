# Fault/mapping interface probes

`grant-double.c` consumes `native-grant.h` and runs without a guest. Run
`bash test-double.sh OUT` on the measurement host. The normal implementation
must return zero; every named mutant must return exactly one. Log files retain
each failed assertion. This models authority and ordering, not instruction cost.

`linux-6.1-grant-unmap.patch` applies after the existing unpinned Linux 6.1
patch to `mm/mmap.c`. It exports a strict whole-VMA helper. Build the kernel
with the existing RISC-V configuration and matching cross compiler, then build
the adapter module against that kernel with `CV_GRANT_TXN=1`. Ordinary builds
omit the transaction ioctl entirely. The platform must still enable raw moves
and load the module with `unpinned=1`; it must not use huge pages.

Build `grant-probe.c` with the plain RISC-V Linux compiler, `-static -O2
-Wall -Wextra -Werror`, and run it as an ordinary guest process. Its parent
context never enters a domain. Five injected failures must preserve the VMA,
contents, arena count and live/retired node counts exactly. Each is followed
by an uninjected success that removes the VMA and retires its ancestor.
The probe also refuses partial geometry, stale mapping IDs, unknown versions,
unsupported map requests, a descriptor inherited into a foreign address space,
and grants spanning split VMAs. Failed requests have no authority effects.

The probe does not exercise descendant dereferences or active transport holds:
those require the later native integration. The interface double includes
descendant and sleeping-revoker controls. For real-MM negative controls, build
private copies of the test module with its retire callback returning zero
without retirement, or move retirement before helper preparation. The first
must fail the successful-unmap live-node check; the second must fail the
injected-error arena/liveness check. Never run those modules in scored builds.

No MAP/PROTECT/REMAP implementation or complete mapping ABI qualification is
claimed. See the [design](../../../docs/design/virtual-capstone/native-grant-transactions.md)
for ordering, supported geometry and remaining integration requirements.
