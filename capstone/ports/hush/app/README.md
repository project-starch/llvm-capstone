# Hush native spawn shell

This port builds BusyBox 1.37.0's hush parser as a noninteractive capability
application. The archive is downloaded beneath `CAPSTONE_TMP_ROOT`; upstream
sources are not vendored. `prepare-sources.sh` pins the archive by SHA-256 and
applies the versioned patch before the shared SDK compiles every target object.

The patch replaces hush's NOMMU fork/vfork process boundaries with
`posix_spawn`/`posix_spawnp`. It retains upstream parsing, expansion, redirection,
function serialization, and positional arguments. External commands run as
children and their exit status determines the pipeline status. Subshells,
pipeline builtins/functions, and command substitution start the same capability
shell using hush's existing NOMMU state serialization. Large heredocs use the
existing reexecution writer mode; the resident shell reaps that child directly.

Internal pipe and saved descriptors have `FD_CLOEXEC`. Parent redirections are
temporary, restore descriptor flags, and close user descriptors that temporarily
occupied a relocated script descriptor. The shell is single threaded. It does
not optimize the final external command into an in-place exec.

## Build

From the repository root:

```sh
source capstone/tests/capstone-test-env.sh
bash capstone/ports/hush/app/build.sh --sdk /path/to/common-sdk \
  "$CAPSTONE_TMP_ROOT/hush-exchange"
```

The fresh output directory contains `hush.dom`, `manifest.json`, the pinned
source tree, `.config`, and build logs. The manifest includes archive, patch,
configuration, image, and SDK identities. The recipe uses the SDK's recorded
linker for BusyBox's intermediate relocatable links and its compiler driver for
the final capability image. It does not strip that image.
It selects upstream's documented `BB_GLOBAL_CONST=` option: declaring mutable
global pointers `const` otherwise allowed a load before initialization in the
capability compiler, producing an untagged null pointer at shell startup.

For a host build and process-semantics checks:

```sh
bash capstone/ports/hush/app/build.sh "$CAPSTONE_TMP_ROOT/hush-host"
python3 capstone/ports/hush/app/tests/host-check.py \
  "$CAPSTONE_TMP_ROOT/hush-host/busybox" "$CAPSTONE_TMP_ROOT/hush-tests"
```

`tests/semantics.sh` and the ordinary `tests/child.c` executable can also be
staged for native qualification. Compile the child with the same SDK and run
the script as `sh semantics.sh ./child.dom`. The tests exercise argv, environment,
status, pipelines larger than the pipe capacity, subshell isolation, serialized
functions, command substitution, redirection restoration, descriptor leaks,
heredocs, and control flow. The host driver additionally checks a 100 KB heredoc
and repeated redirection failures. Host success alone is not native execution
evidence.

The host and native exchange/direct fixtures pass for the artifacts identified
in the [qualification record](results/native-task.json). Raw logs remain in the
external evidence directory named by that record.

`tests/run-native.py` uses the shared QEMU runner and lock. Supply `--shell`,
`--child`, `--driver`, `--qemu`, `--images`, `--firmware`, and a fresh `--work`
directory under `CAPSTONE_TMP_ROOT`. It hashes the platform and test inputs
before the run and writes `result.json`. Its boot settings explicitly enable
the pinned shell and null-device grants; the remaining filesystem authority is
the staged cwd. It additionally qualifies a 100 KB heredoc and repeated failed
redirection cleanup in the native task.

## Authority and supported scope

For the native build, the self-execution name is `/bin/sh`. The trusted native
loader must map that exact name to this pinned capability image and inherit
that file grant into descendants. This port grants no directory, installs no
Linux shell fallback, and never invokes a scalar launcher. Ordinary executable
and data access remains bounded by the calling task's existing grants.

The configured scope is noninteractive commands, scripts explicitly read by
hush, pipelines, functions, substitutions, heredocs, and ordinary control flow.
Interactive operation, job control, caught signal traps, and Bash extensions
are disabled. `exec COMMAND` requires the still separate in-place native exec
service and currently fails; it is not emulated by spawning a replacement PID.
The `command`, `kill`, `umask`, `ulimit`, and `times` optional builtins are not
included. Background commands require readable `/dev/null` authority for their
default stdin (`dom_null=1` supplies the exact device grant); failure to acquire
it is an error. Background statuses are retained even if a foreground pipeline
reaps those children before `wait`. Script-as-executable fallback
is not added to the native spawn service. These limits must not be described as
complete POSIX shell support.

Serialized subshell state, command substitutions, arguments, and environment
must fit the native spawn request's 64 KiB limit. Oversized requests fail with
`E2BIG`. A large heredoc read from a script can stream through its pipe without
embedding the entire script inside a serialized command substitution.

Native qualification and full Perl qualification are separate gates. An
upstream shell and a successful compilation do not by themselves establish
that the benchmark's complete subprocess tree ran.
