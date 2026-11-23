# The php engine-repros on kit-p1v, the P1 twin (2026-11-23)

These are the QEMU verdicts that the P1 board run is predicted from: the same image bytes (`images.sha256`) under dev
1a051da6e82b's launcher and its module built flat for ids55. The QEMU is capstone-qemu 0542c5556c with
`x-capstone-ids=55` and `x-capstone-addr=sv39`, and without exact bounds.

The readings, their comparison with the earlier bundles, the identities and the board pre-registration are in
`docs/ref/nested-allocator-workloads.md`, sections "Readings on kit-p1v, the P1 twin" and "Pre-registration: item 5
on a P1 board". Account paths in `inputs.json` are written `<hdd>/`. Raw guest output is not committed.
