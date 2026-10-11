# Positive control for the `sublet-malloc` arm

The `sublet-malloc` arm builds the corpus against wmem **as released** (`WM_VARIANT=reference`):
none of the port's patches, so the only lifetime event Sublet sees is a `g_free` -- every
`g_malloc` is a region and every `g_free` revokes it, as the Sublet heap does under a stock
program. Every case in the corpus is predicted to COMPLETE there, because no reset returns the
stale object's block to the system. A column of completions proves nothing unless this build can
also fault, so this directory holds the one case that must:

* `90_ctl_jumbo_freed_by_reset` allocates an object larger than a BLOCK_FAST block from the packet
  pool. wmem gives it its own `g_malloc` (a jumbo), and `wmem_free_all` on the packet pool frees
  every jumbo with `g_free`. The stale read after the reset must FAULT in mode 1 (revoked) and
  COMPLETE in mode 0 (the same build, nothing revoked).

Build and run it exactly as the corpus, pointing both at this directory:

    cmake --preset capstone-domain -S <wmem port> -B <out> -DWM_CHUNKS=OFF -DWM_VARIANT=reference \
        -DWM_CORPUS_DIR=<this directory>
    shared/run-defects.py <results> --domain-build <out> --linux-build <guest> \
        --corpus <this directory> --modes spatial,sublet

`shared` is a link to the corpus's own driver, so the control runs the same seam.
