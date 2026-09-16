# libc-test patches for a capability target

Each patch removes one assumption of flat memory from a TEST, never from musl:
an object read past its own bounds, or a pointer sent through an integer and
back. On a flat target both are harmless and the test passes; under exact
bounds and tags the test faults before it measures anything. CheriBSD carries
the same class of patch for its libc-test port.

Rule for adding one: the patch may change how the test reaches memory, not what
it checks. A test that stops checking something is a different test.

Applied by fetch-libc-test.sh after checkout, idempotently: the tree is reset to
the pinned commit first, then every patch here is applied in name order.
