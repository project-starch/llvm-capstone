# Check-then-copy race experiment (native checked copies)

Go/no-go for kernel-side checked copies on the native path, in the real guest
kernel. Module build `make CV_RACE_TEST=1` only (`CV_RACE_TEST=1 build-adapter.sh`);
the production module compiles none of it.

* Thread A issues a native `read()` (2 pages) from a pipe (`pipe-fill`) into a
  capability buffer.
* Thread B is a real second domain thread of the same context (its own start
  frame, built as `thread.c` builds one; it runs CREVOKE, CDERIVE, stores and
  the SDK's MAP/UNMAP services). The module runs B on the single hart at one
  deterministic point of A's checked copy: `hold` (after the blocking issue),
  `between` (between chunks), `fpre`/`fpost` (A's fault resolution, before and
  after its `handle_mm_fault`), `window` (a reschedule that became pending
  inside the exclusion window; taken at its end when the window excludes
  interrupts, inside it when the control removes the exclusion).
* Variants: `obj` (B revokes the buffer's object and derives a sentinel object
  over the next page), `objgc` (as `obj`, plus the identity-table collection
  that recycles dead identities), `map` (B munmaps the buffer's mapping and maps
  a new one), `nop` (B runs and changes nothing).
* Page 2 of the buffer is `res`ident, `absent` (demand fault) or `cow` (made
  present by a read fault, so A's write takes a copy-on-write fault).
* Controls, module parameters of the race build: `rt_norecheck` (only the first
  chunk is authenticated), `rt_noexcl` (no irq/preempt/pagefault exclusion),
  `rt_dummy` (prefault by a dummy store), `rt_nohold` (no hold on the mapping).

`scenarios.txt` is the pre-registered list, `judge.py` its expectations (every
safe scenario must pass; every control must produce its violation, otherwise the
control is void). `build.sh` and `run.sh` reproduce the run on a build host.
What is not covered: SMP, device DMA, signals, a scheduler-chosen (not
hook-chosen) interleaving, and an identity actually reused with a new generation.


## Copyable-only qualification

Both `native_fast=0` and `native_fast=1` pass all 41 scenarios (24 safe, 17
negative controls) on the matching unpinned single-hart model. No ownership
fields or legacy instructions are imported. The test build adds deterministic
hooks; the production module contains no `rt_*` module parameters.

Build with `bash build.sh OUTPUT` after selecting `KERNEL_BUILD`, `CROSS_COMPILE`
and `CAPSTONE_NATIVE_CC`. `run.sh` forwards to the native integration runner in
`race` mode; pass `--module OUTPUT/adapter/module/capstone_vm.ko --probes OUTPUT`
and the common model/kernel/firmware/output options described in
[the integration instructions](../native-integration/README.md).
Repeat with fresh outputs for `--fast 0` and `--fast 1`.
