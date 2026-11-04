# Checked copy reference experiment

This is a hosted test of the shared `cu_copy` loop, linked to the native
context double. It is not a Linux module, guest scheduling or instruction
attempt measurement. Run on the build host:

```sh
source capstone/tests/capstone-test-env.sh
bash capstone/runtime/virtual/tests/checked-copy/run.sh "$CAPSTONE_TMP_ROOT/copy-check"
```

The blocked copier, pipe synchronization, reuse cases and exact sentinel
oracle adapt the earlier `inline-io-contract.c` experiment. A cooperating
pthread revokes and reuses an object or mapping between page chunks and in
fault resolution. The transport is a private kernel-buffer analogue. The
reference lifetime generation is synthetic; authority is obtained through
`cn_authorize`, never from a caller-supplied address. Partial copy and
persistent-fault cases check accounting and bounded retries.

A mapping hold returns logical unmap success immediately, retires authority,
and defers physical unmapping. MAP_FIXED_NOREPLACE verifies that the old VA
remains occupied until release. The COW cases use a private file mapping;
the replacement object's writes leave the file unchanged. An injected short
copy models the kernel's fault-disabled copy return; the resolver itself
writes no byte. This does not claim that host Linux's COW handler is under
test instrumentation, or that guest handle_mm_fault is qualified.

Negative controls remove the repeated check, allow a revoker after validation
inside the copy window, perform a dummy resolver store, or remove both the
check and hold to permit real MAP_FIXED reuse. Every variant must fail the
same sentinel oracle. Disabled exclusion is injected on the second chunk so
that the first chunk cannot mask the missing guard.

The kernel adapter and CSCHECK ISA remain experimental and default off.
Actual native-task plumbing, single-hart guest retries, attempt budgets and
both-host nine-gate / quick / full census qualification are required before
integration. The existing held-exchange bridge remains the production path.
