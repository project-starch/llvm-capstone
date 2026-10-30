# 20261028-host-asan

Host ASan at the pin, for the 30 cases of this corpus. It is liveness evidence and the
reading that places each case in this corpus rather than the sibling one -- not a protection column.

- `matrix.tsv` -- one row per case: the detector build's reading, the stock-allocator build's
  reading, how the block was released, and the first frame of the report's access stack.
- `inputs.json` -- the two builds, what differs between them, and the three deviations this host's
  toolchain forced.

With Zend's allocator as shipped, 0 of 30 still produce a temporal report here.
