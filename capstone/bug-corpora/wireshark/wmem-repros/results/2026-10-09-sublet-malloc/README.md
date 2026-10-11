# wmem-repros: Sublet ONLY as the system allocator (`sublet-malloc`), 2026-10-09

**Question.** Column 2 of the per-bug table: with wmem left exactly as Wireshark 4.6.8 ships it
and Sublet as the system allocator under it -- every `g_malloc` a region, every `g_free` a revoke --
which of the 22 cases does Sublet catch?

**Build.** The wmem port with `-DWM_VARIANT=reference -DWM_CHUNKS=OFF`: the corpus's domain
programs compile `source-reference`, the pinned release with none of the port's patches (0 of the
port's hook markers in its six units, against 96 in `source-ported`), so no reset ends an epoch
and no chunk is a region. Run in mode 1 by `shared/run-defects.py --modes sublet`, which judges it
against each case's `sublet-malloc` arm (the build's `WM_VARIANT` selects the arm).

**Result: 0 of 22 caught, every one as pre-registered (f7adf03a1009).** All 13 temporal cases complete:
the stale object lives in the packet pool's first BLOCK_FAST block, which `wmem_free_all` keeps
and reinitialises, or (case 12) on the epan BLOCK allocator's free list inside a live block -- no
`g_free` happens before the stale access. All 9 spatial cases complete: the crossing leaves its
chunk and stays inside the block, one `g_malloc`.

**The completions are misses, not a dead mechanism.** `controls/sublet-malloc/90` is a jumbo the
reset DOES hand to `g_free`: in the same build options it faults at the labelled read probe with
cause 24 in mode 1 and completes in mode 0. The control's first boot was an infrastructure flake
(QEMU stopped before the guest login; the runner exited 75, no reading); the second boot is the
record. `result-lines.txt` carries every row and each image's hash.
