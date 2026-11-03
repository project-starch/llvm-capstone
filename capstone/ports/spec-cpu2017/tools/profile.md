# Profiling SPEC on the virtual profile: instruction attempts, not seconds

## What it measures

`tools/profile.sh` runs one benchmark at one size in one arm exactly as `tools/run.sh` does
(runcpu validates it in a freshly booted guest), with two differences: the guest's QEMU carries
capstone-qemu's `contrib/plugins/capprof.c`, and every guest invocation runs under `profrun`
(`tools/capprof-validation/profrun.c`), which opens a capprof window, forks and execs the
invocation, waits, and closes the window. The plugin counts, per instruction address, how many
times the guest ATTEMPTED that instruction inside the window: an instruction that faults and is
resumed counts each time, a translated-but-not-executed block counts nothing. `prof-report.py`
attributes every counted address to a layer (program, launcher, profrun, kernel, module,
firmware, other) and to the containing function, and checks the instruction bytes against the
image they are attributed to. `prof-compare.py` gives the signed per-function difference of two
runs: the same arm twice (noise), or two arms of one benchmark (where the instructions go).

An attempt is neither a retired instruction nor a cycle. A 16-byte capability load and an
`addi` each count one, and software that exists only because the processor is emulated (the
supervisor quantum's round trips, the module's page pinning) counts as much as anything else.
The counts therefore rank the SOFTWARE STACK's instruction expansion -- compiler output, libc,
SDK runtime, launcher, module, kernel work done for the program -- which is what a software
optimization changes. They do not rank hardware cost; a hardware claim needs the cache model
(the 2026-10-20 cachesim lane) or silicon.

## Why counts, and what they buy

A SPEC test input takes minutes to hours in QEMU (perlbench 3.4 h, x264 3.6 h, the kit's
qualification of 2026-10-17), and a timing must be repeated on pinned, quiet cores. Attempt counts
are a property of the instruction stream, so a run's wall-clock and the host's load do not
change them. Every benchmark and arm runs at once (`tools/profile-batch.sh`), unpinned.

Reproducibility: the guest runs with `-icount shift=0` (guest time follows the instruction
count, so timers and the supervisor quantum, a QEMU_CLOCK_VIRTUAL timer, fall at the same
instructions every run), address-space randomization off, syslogd/klogd stopped. The validation
programs then count IDENTICALLY run to run in their user-level windows; what still varies is
guest-kernel work driven by host-timed events (virtio completions, the SSH session), which
lands inside a window or outside it. On the validation set that was 0 in five of seven windows
and tens of thousands of kernel attempts in a window that pins pages (`vt3`/`vt4`, 2026-11-02:
355,638 vs 403,588 in the fault test, all of it kernel). Report that floor beside any difference
near it; `prof-compare.py` on two runs of the same arm measures it for a benchmark.

## Arms and fairness

An arm (`config/profile-arms.example.json`) names what it measures: for the Capstone arm the SDK
(so the compiler, musl-capstone, the heap and runtime policy), the module variant and the CPU
properties; for the plain arm the compiler driver built from the same llvm-capstone source with a
pristine musl 1.2.5 (`config/riscv-plain.cfg`). Both arms compile the same sources with the same
src.alt packages and the same -O2 and portability flags, and both use the F and D instructions
with the lp64 convention. They differ in the libc (musl-capstone's patches), the SDK runtime, the
launcher and module, and the capability code the compiler emits: the comparison is the whole
software stack of the virtual profile against a plain static Linux program, not the capability
ISA alone. A function present in one arm only (an SDK routine, a libc replacement) appears in
the comparison as unmatched, not hidden.

The window is system-wide: everything the guest executes between the markers counts, including
the kernel's work for other processes. The guest is quiesced (one vCPU, no daemons but dropbear),
and the report's `kernel` and `other` classes are read with that in mind.

## What a run records

`OUT/record.txt` (the arm's JSON, platform, icount, memory, share model), `inputs.sha256` (QEMU,
images, adapter, module, plugin, profrun, System.map), `qemu-command.json` (the exact command
line, CPU properties included), `module-sections.txt` (the module's section addresses, the
kernel's command line, the module parameters, ASLR state), `invocations/NNNN.json` (route, argv,
cwd, executable hash, VM identity, guest exit record, window index), `image-records.txt` (the
launcher's entry and code range per invocation), `capprof/window-N.tsv` (the raw counts with the
plugin's footer), `spec/` (runcpu's log and reports, the run directory's command and output
files), `rundir.sha256`, `result.json` (`tools/verdict.py`: PASS only when runcpu validated the
outputs), `windows.json` (every guest invocation has a complete, valid window), `profile.json`
and `profile/window-N.json` (the attribution). The build's `build-record.txt` holds the compiler
and SDK hashes and the executables'. A run whose verdict is not PASS or whose windows are
incomplete exits 1.

## Validation of the instrument

`tools/capprof-validation/vt-run.sh TAG` boots the platform and runs the fixed programs
(`vt.c`, `alias.c`): loops with known trip counts in a domain and as a plain program, a 50M-loop
window that crosses quanta, a prefault and a fault test whose retried stores must be counted
twice, and an alias test (the same bytes at two addresses must be two records). `vtcheck.py`
fails unless every check passes. Run it on every new platform before a measurement.

## Long benchmarks

perlbench, x264, mcf and xz test inputs run for 45-220 minutes. A slice "first N attempts"
compares unequal work between arms (the arm with more instructions per iteration gets fewer
iterations), so it is not used. The honest reduction is a smaller workload that finishes
normally and validates: SPEC's test input where it is short, otherwise a reduced input
labelled as such. (Open.)
