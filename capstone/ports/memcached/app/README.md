# Full memcached on virtual Capstone

This is the complete memcached 1.6.45 server with four workers and libevent
2.1.12, rebuilt for the virtual application SDK. Linux is trusted and schedules
the same-process workers. Application code runs in virtual C mode; workers
share one virtual address space and lifetime namespace. This is not the
allocator seam under `../allocators`.

The original virtual port started from `memcached-app` at `c839e7294e23`.
This review slice preserves current dev’s additional slab/cache adapters,
corpus fixtures and pre-registered safety predictions.
The physical build/oracle/safety recipes and their three heap arms remain
available. The virtual recipes use the common revoking heap and actual musl
pthread/TLS/mutex/condition support. They need no new processor instruction,
kernel module change, Linux-core patch or firmware change.

## Build and run

Source `capstone/tests/capstone-test-env.sh` with the qualified Capstone
compiler configured. Use a fresh directory under `/tmp/capstone`:

```sh
bash capstone/ports/memcached/app/host/build-virtual.sh /tmp/capstone/memcached-virtual
```

This rebuilds musl/runtime, libevent, the server, native controls, worker-marker
and safety variants, and the pthread contract. `CAPSTONE_APPLICATION_PROFILE`
is set to `virtual`; physical objects cannot be converted by relinking.
The dependency gate runs upstream native libevent tests, cross-config review,
link failure controls and the existing dependency-symbol scope check.

```sh
python3 capstone/ports/memcached/app/host/run-virtual.py \
  --build /tmp/capstone/memcached-virtual \
  --adapter /tmp/capstone/virtual-adapter \
  --qemu capstone/capstone-qemu/build/qemu-system-riscv64 \
  --images /tmp/capstone/virtual-capstone-abi/images \
  --cross-cc "$CROSS_COMPILE"gcc \
  --pthread /tmp/capstone/memcached-virtual/pthread.dom \
  --work /tmp/capstone/memcached-virtual-gate
```

`--adapter` contains `capstone-vexec`, `capstone-job` and
`module/capstone_vm.ko`; see the [runtime build instructions](../../../runtime/virtual/README.md).
The gate boots an ephemeral one-hart Linux guest. Its device permissions and
unprivileged server launch are test setup, not an installed system policy.

## Qualification

The 53-check record below is historical and predates current dev’s expanded
corpus. The [fresh review result](results/virtual-review-result.json) passes **73/73**
checks on the extracted dev-based stack. It includes all 20 fixtures with a
`sublet` prediction in the current manifest, rather than stopping at fixture
10. Normal, safety and worker-marker servers and native controls were built
afresh; the Capstone libevent archives were reused, with their hashes and
build scope recorded. Upstream libevent tests were not rerun.
The virtual profile here uses the outer revoking heap.

### Sublet inside the slabs and the object cache

Patch 0006 (`-DMC_CAPSTONE_SUBLET`) closes the nested-allocator gap with two
instructions in memcached's own allocators: `slabs.c` and `cache.c` hand out
every chunk and object as a CDERIVE child lifetime, and `slabs_free` and
`cache_free` revoke it (CREVOKE). There is no adapter. The patch header says
how the slab mover, chunked items and page moves are covered and what it
gives up. `host/build-sublet.sh` (run by `build-virtual.sh`) builds its oracle
and safety images under `$MC_WORK/sublet/`, and gates that both carry the two
instructions and the plain images none. `run-virtual.py --arm sublet` runs the
same gate on them and judges the fixtures by the `lifetimes` predictions in
`host/safety-expect.txt`, registered before the first build. The physical
`slabsublet` arms, the allocators port's ledger inside memcached on the Sublet
heap, were removed on 2026-10-11 with that heap's lend API; their predictions
stay in `host/safety-expect.txt` as the record.


The [historical virtual result](results/virtual-result.json) passed **53/53 checks** and
records exact input hashes. The existing text/meta-protocol harness exercises one
connection and eight concurrent connections, comparing the normalized bytes
against native memcached. It checks TERM/USR1 termination, stderr, pointer
width, all four worker markers, and negative controls for changed values,
changed CAS results and missing worker evidence. Normal output is compressed
for serial transport without changing the comparison bytes.

In that historical run, ten worker-context fixtures were judged by the existing common safety
classifier against the unchanged physical **Sublet** predictions. Fixtures
2/3/7/8 deny bounds violations; 4/5/6 deny stale/reused/double-free access.
Fixtures 9/10 still expose slab-neighbour and slab-reuse access: slab items
are carved from a larger allocation without individual capability lifetimes.
Virtual addressing alone does not close that nested-allocator gap.

The [pthread result](../../../runtime/virtual/pthread-result.json) covers TLS,
tagged joins, mutex/conditions, timeout, independent blocking I/O, private
epoll events, shared-TLS explicit-context transport and repeated joins.
Both libc's epoll conversion buffer and the launcher's message views/iovecs
are per invocation. Shared buffers permitted foreign event or payload
copy-back across connections. The
[epoll controls](../../../runtime/virtual/pthread-epoll-controls.json) and
[message controls](../../../runtime/virtual/pthread-host-controls.json)
expose those errors without relaxing the protocol oracle.

The preserved [physical result](results/2026-10-01-qemu-safety/README.md)
is provenance, not a virtual result. Raw guest logs and binaries stay outside
the repository. The reusable pthread bridge is reviewed separately from this application port.

## Limits

This qualifies the configured server/workload on QEMU with one hart. Extstore,
proxy, TLS, SASL and documentation builds retain the physical port's disabled
configuration. Full POSIX cancellation/signals, fork, shared tagged pages,
file-backed mappings, partial unmapping, migration, SMP and RTL/FPGA remain
outside this result. The runtime's compact-bounds conformance limitation also
remains; these gates are not a complete proof of spatial/temporal safety.
