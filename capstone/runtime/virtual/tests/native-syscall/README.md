# Native syscall executable contract

This directory exercises the imported shared syscall dispatcher with a hosted
context double. See the [integration boundary and source revisions](../../../../docs/design/virtual-capstone/syscall-import.md).
The checked-copy provider is synthetic. This branch does not yet have the source
lane's native kernel adapter, checked-access ISA or native SDK entry selected.

```sh
source capstone/tests/capstone-test-env.sh
bash capstone/runtime/virtual/tests/native-syscall/run.sh "$CAPSTONE_TMP_ROOT/syscall-results"
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
| OFFSET | authenticated exchange offset retained |

Software counts are not instruction attempts. No Linux syscall, VFS permission
hook, seccomp filter, real capability copy or application oracle is qualified
by these tests. The source lane's native guest race probes and full program gates remain
required after its kernel/model providers are adapted to this branch.
