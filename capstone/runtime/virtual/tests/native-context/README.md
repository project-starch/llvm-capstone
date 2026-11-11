# Hosted context reference

The reference provides protected-frame, namespace and epoch callbacks to the
imported syscall and checked-copy contracts. Its registers and authority checks
are synthetic; it neither implements a kernel context switch nor enables a
native task or protected fork. Context-slot movement is test bookkeeping, not
linear ownership of application memory. The [integration boundary](../../../../docs/design/virtual-capstone/syscall-import.md)
records the immutable upstream revision and unimported dependencies.

```sh
source capstone/tests/capstone-test-env.sh
bash capstone/runtime/virtual/tests/native-context/run.sh "$CAPSTONE_TMP_ROOT/context-results"
bash capstone/runtime/virtual/tests/native-context/run-peers.sh \
  "$PWD" "$PWD/capstone/runtime/virtual/unpinned/lifecycle" \
  "$CAPSTONE_TMP_ROOT/context-peers"
```

Use fresh output directories. The first runner checks the reference under
ASan/UBSan (non-recovering) and requires six broken variants to fail. The second
composes it with this checkout's mapping transaction and tag-page lifecycle
fixtures. Neither runner measures guest instructions or qualifies actual Linux
syscalls, memory ordering, scheduling, or tag hardware.
