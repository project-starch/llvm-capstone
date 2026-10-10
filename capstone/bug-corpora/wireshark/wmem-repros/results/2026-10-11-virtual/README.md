# wireshark/wmem-repros on virtual Capstone -- 2026-10-11

The first run of the wmem port's patch 0001 (every object of `block` and `block_fast` a child
lifetime of its block, `CDERIVE`/`CREVOKE`). Source: commit `0e22fc33d297`
(`nested/wireshark-sublet`), built with the `capstone-application` preset against a virtual SDK,
`WM_SUBLET` off for `virtual-malloc` and on for `virtual-nested-pools`. One VM, virtual profile with
exact bounds, the Sublet-lifetime QEMU; the hashes are in each bundle's `inputs.json`.
Verdicts are the shared judge's (`tools/verdicts.py`), derived into the cases by
`tools/derive-verdicts.py`.

| arm | configuration | controls (uaf-chunk, uaf-jumbo, bounds-chunk) | cases | time |
|---|---|---|---|---:|
| `virtual-malloc` | `virtual-wmem` | complete, fault, complete -- as declared | MISSED 22 / 22 | 21 s |
| `virtual-nested-pools` | `virtual-wmem-pools` | fault, fault, fault -- as declared | CAUGHT 22 / 22 | 22 s |

Every protected fault lies in the labelled probe the case uses: `wm_probe` for the reads,
`wm_write_probe` for the writes of cases 16, 17, 18, 20 and 21. Cases 0-12 (temporal) fault with
cause 25 (invalid lifetime); cases 13-21 (spatial) with cause 28 (out of bounds). Every unprotected case
reached its ready mark and completed.

## The allocator still works

The port's directed replay -- 1,661 events over all four allocators: 1,000 allocations, frees,
in-place and moving reallocs, jumbo objects past both block sizes, resets, collections and
destructions -- ran as a Capstone process from both builds. Its report is byte-identical to the
native one:

    native                 WM completed=1661 allocs=1000 checksum=15475871146958098728 system=800 peak=33
    virtual, WM_SUBLET=OFF WM completed=1661 allocs=1000 checksum=15475871146958098728 system=800 peak=33  exit 0  205 s
    virtual, WM_SUBLET=ON  WM completed=1661 allocs=1000 checksum=15475871146958098728 system=800 peak=33  exit 0  206 s
    report sha256 (all three) 74723e2a36633d1a94f154ee81f6e39053440071caf33a79cc69d501e9185d06

The replay checks every live object's payload before each free, realloc and reset, so an object
the patch had clobbered, or a pointer it had bounded too tightly, would have failed it. The
replay's report is not part of the verdict bundles.

N = 1 per cell. The builds took 6 s each and the VM 18 s to boot.
