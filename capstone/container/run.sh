#!/usr/bin/env bash
# Run a command inside the Capstone build image against the HOST source tree.
#
#   ./run.sh                                  # interactive shell, env already sourced
#   ./run.sh capstone/container/setup.sh      # the one-shot bring-up
#   ./run.sh bash -c 'source capstone/tests/capstone-test-env.sh && "$CAPSTONE_CLANG" --version'
#
# Everything the project needs is reachable from ONE mount, because capstone-qemu and
# caplifive-buildroot are submodules INSIDE llvm-capstone. capstone-test-env.sh derives
# CAPSTONE_QEMU_BINARY, CAPSTONE_BUILDROOT_DIR and CAPSTONE_LLVM_BIN from
# CAPSTONE_REPO_ROOT, so with the repo at /work/llvm-capstone every default resolves and
# no script needs an override.
#
# The flags below are load-bearing:
#
#   --userns=keep-id
#       Runs as your own uid instead of container-root. Two reasons, either sufficient:
#       (a) build outputs in the bind mount come back owned by you rather than by subuid
#       101000, which you could not delete without `podman unshare`; (b) Buildroot
#       REFUSES TO RUN AS ROOT and aborts its configure step.
#
#   -e HOME=/home/builder
#       With keep-id the image's /etc/passwd has no entry for uid 1000, so HOME would be
#       unset. capstone-test-env.sh then resolves CAPSTONE_QEMU_LOCK to
#       "/.capstone-locks/qemu.lock" and its mkdir fails -- which happens at SOURCE time,
#       so every script that sources it dies before doing anything. The directory is
#       bind-mounted from container/home so the lock, the ccache and the Buildroot
#       download cache persist between runs.
#
#   -v ...container/tmp:/tmp/capstone
#       CAPSTONE_TMP_ROOT is where every test script writes artifacts and logs. A
#       container-local /tmp would discard them on exit, which is exactly the evidence
#       you want after a failing run.
#
#   -e CAPSTONE_REPO_ROOT=/work/llvm-capstone
#       Set explicitly rather than left to capstone-test-env.sh's own derivation. That
#       derivation is BASH_SOURCE-based and has silently produced "/home" before (see the
#       zsh note at the top of capstone-test-env.sh); pinning it removes the failure mode.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"          # .../llvm-capstone
IMAGE="${CAPSTONE_IMAGE:-capstone-build}"

[ -d "$ROOT/llvm" ] && [ -d "$ROOT/capstone" ] || {
  echo "run.sh: $ROOT does not look like the llvm-capstone root" >&2; exit 2; }

mkdir -p "$HERE/home" "$HERE/tmp"

# -i ALWAYS: without it podman does not forward stdin, so `./run.sh python3 - <<EOF`
# silently runs an empty script and exits 0. -t only when we really have a tty, so this
# stays usable from scripts and from a non-interactive agent session.
TTY_FLAGS=(-i)
[ -t 0 ] && [ -t 1 ] && TTY_FLAGS=(-i -t)

# The project's GitHub repos are PRIVATE and this machine authenticates with
# `credential.helper = store`, i.e. a token in ~/.git-credentials. That token is
# deliberately NOT in the container: source fetching happens on the host via
# capstone/container/fetch-submodules.sh, because a container that builds an entire
# Buildroot tree runs a great deal of third-party code. CAPSTONE_GIT_CREDS=1 opts in.
# The PHP 5.0.0 corpus tree lives OUTSIDE llvm-capstone, so the single repo mount does
# not reach it. Mounted READ-ONLY and only when present, so nothing here breaks on a
# machine that does not have the corpus: the ports that need it check for /corpus and say
# so, rather than failing with a confusing "no such file".
CORPUS_FLAGS=()
_corpus="${CAPSTONE_PHP_CORPUS:-$(dirname -- "$ROOT")/case-studies/php-5.0.0-bug-corpus}"
if [ -d "$_corpus" ]; then
  CORPUS_FLAGS=(-v "$_corpus:/corpus:ro")
fi

CRED_FLAGS=()
if [ -n "${CAPSTONE_GIT_CREDS:-}" ] && [ -f "$HOME/.git-credentials" ]; then
  CRED_FLAGS=(-v "$HOME/.git-credentials:/home/builder/.git-credentials:ro"
              -e GIT_CONFIG_COUNT=1
              -e GIT_CONFIG_KEY_0=credential.helper
              -e GIT_CONFIG_VALUE_0=store)
fi

# No args -> interactive shell with the project env already sourced, which is what you
# want 90% of the time and is easy to forget by hand.
if [ $# -eq 0 ]; then
  set -- bash --rcfile /dev/stdin <<<'source /work/llvm-capstone/capstone/tests/capstone-test-env.sh 2>/dev/null; cd /work/llvm-capstone'
fi

exec "$HERE/pod.sh" run --rm "${TTY_FLAGS[@]}" \
  --userns=keep-id \
  "${CRED_FLAGS[@]}" \
  "${CORPUS_FLAGS[@]}" \
  -v "$ROOT:/work/llvm-capstone" \
  -v "$HERE/home:/home/builder" \
  -v "$HERE/tmp:/tmp/capstone" \
  -w /work/llvm-capstone \
  -e HOME=/home/builder \
  -e CAPSTONE_REPO_ROOT=/work/llvm-capstone \
  -e CCACHE_DIR=/home/builder/ccache \
  -e CARGO_HOME=/home/builder/cargo \
  -e JOBS="${JOBS:-$(nproc)}" \
  "$IMAGE" "$@"
