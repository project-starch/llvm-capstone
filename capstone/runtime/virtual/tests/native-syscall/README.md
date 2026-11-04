# Native syscall executable contract

This directory provides a default-off syscall dispatcher double. See the
[interface and limitations](../../../../docs/design/virtual-capstone/native-syscall.md).
It links N's protected context implementation directly. U is represented by
the required checked-copy callback; absence returns ENOSYS before issue.

On a build host, in a lane-owned source export:

```sh
export CAPSTONE_TOOLCHAIN_CHECKED=1
export CAPSTONE_TMP_ROOT="$HOME/lanes/progS/tmp"
bash capstone/runtime/virtual/tests/native-syscall/run.sh "$HOME/lanes/progS/syscall-results"
bash capstone/runtime/virtual/tests/native-syscall/check-sdk.sh \
  "$HOME/lanes/progS/sdk-results" "$CAPSTONE_CLANG" \
  "$CAPSTONE_LLVM_BIN/llvm-objdump" "$CAPSTONE_LLVM_BIN/llvm-nm"
```

Outputs must be fresh. Keep logs and hashes; remove build subdirectories after
use. The public descriptor inventory has 106 unique entries. Synthetic tests
cover input snapshots, aliasing, flat RV64 layouts, vector reference import,
zero lengths, overflow, bounds/rights, stale epochs, namespace/owner refusal,
revocation after blocking and between chunks, short I/O and effect-aware
restart. The runner must reject every mutant for its named oracle:

| Mutant | Required failure |
|---|---|
| PROC | proc-memory open refusal |
| INHERITED | inherited descriptor refusal |
| REPLACEMENT | descriptor identity replacement refusal |
| OVERLAP | pathname/output snapshot isolation |
| HIDDEN | hidden-pointer command refusal |
| RECHECK | recycled object sentinel preservation |
| TAGS | scalar output tag clearing |
| IOV | original vector used for copy-out |
| RESTART | no restart after effects |

Software counts are not instruction attempts. No Linux syscall, VFS permission
hook, seccomp filter, real capability copy or application oracle is qualified
by these tests. The existing native-dom guest probes and full program gates
remain required, as do F's transaction and L's lifecycle providers.
