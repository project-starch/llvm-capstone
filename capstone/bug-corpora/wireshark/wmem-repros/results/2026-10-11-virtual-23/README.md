# wireshark/wmem-repros on virtual Capstone, 23 cases -- 2026-10-11

The run of `../2026-10-11-virtual-22/` repeated with case 22 (`c702b44a01`, a double free into the
block allocator), which reached this branch from dev. Source: commit `46e2a5d0dd84`
(`nested/wireshark-sublet`), the same virtual SDK, QEMU and VM setup; hashes in each bundle's
`inputs.json`. These are the bundles `corpus.json` names under `verdict_bundles`.

| arm | configuration | controls (uaf-chunk, uaf-jumbo, bounds-chunk) | cases | time |
|---|---|---|---|---:|
| `virtual-malloc` | `virtual-wmem` | complete, fault, complete -- as declared | MISSED 23 / 23 | 21 s |
| `virtual-nested-pools` | `virtual-wmem-pools` | fault, fault, fault -- as declared | CAUGHT 23 / 23 | 22 s |

Cases 0-21 read exactly as in the first run. Case 22 faults with cause 25 (invalid lifetime) in
`wmem_block_sublet_free`, a fault site the case declared from source before this run: its second
`wmem_free` finds the block by address and revokes a child the first free had already revoked.
Unprotected, the double free completes.

N = 1 per cell.
