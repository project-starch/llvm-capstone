# The port's examples and the census probes on the virtual profile, 2026-11-11

The first run of the port's own programs as capability applications (preset
`capstone-application`), taken after `src/virtual/node-derive.c` and
`src/virtual/bucket-derive.c` moved to the one-operand CREVOKE
(`capstone_cap_release` on the node's issued child and on the bucket piece).
Two boots of the QEMU that implements it, pinned module, `-cpu` extra
`x-capstone-test-csrs=true`, every program its own `capstone-vexec` process.
9 of 9 rows read as expected. No earlier record of these programs on this
profile exists, so there is nothing to compare against but the expectation
each program states.

    matrix.tsv    one line per program: build, expected, exit, the line it printed
    inputs.json   image, platform and compiler hashes

What the rows are:

- `bucket-geometry` with `APR_HAS_MMAP` 1 and 0, and `two-levels`
  (`../../../probes/`, built with the SDK's `capstone-cc` from the tree the
  bucket census prepares). The allocation size does not move with
  `APR_HAS_MMAP` here either (224 both ways; 128 on the host, where a pointer
  is 8 bytes), both recycling levels reissue the same address, and the
  interposed `free()` counts zero.
- `allocator-example` and `bucket-example` of the stock build
  (`APRP_SUBLET=OFF`): `ALLOCATOR_EXAMPLE apr PASS pointer_bytes=16` and the
  LIFO reissue, exit 0.
- The same two of the nested build (`APRP_SUBLET=ON`) as the port builds
  them. They select mode 0, which only the stock adapters accept, and the
  nested adapter refuses it by name (`506`, exit 1). That refusal is the
  adapter's own rule and is what keeps a mis-paired build from running
  unprotected.
- The same two sources with the mode argument set to 1, linked against the
  nested library: the examples' whole sequence under the nested adapters. A
  destroyed pool's node comes back as the next pool (`same_node=1 reuses=1`),
  the bucket freelist reissues LIFO (`reissues=2 files=2`), exit 0. So an
  issue after a one-operand release hands out a usable child over the same
  bytes, at both levels.

A nested image with the bucket allocator holds three CREVOKE instructions:
the copy of `capstone_cap_release` in each of the two adapters, and one in
the SDK's runtime that the stock build holds too. All three have the form
`rd = x0, rs2 = x0`; none has the two-operand form.
The corpus readings on the same adapters are with the corpora
(`bug-corpora/httpd/*/results/2026-11-11-virtual/`).
