# gcc's port v2 (CFP-B5TL1-g3 v2): a fail-stop CRENEW, its build, fixture and twin runner (nested lane, 2026-11-26/27)

These are the instruments behind two sections of `capstone/docs/ref/nested-allocator-workloads.md`:
- "Pre-registration: CFP-B5TL1-g3 v2, gcc's port with a fail-stop CRENEW, on b42c8c644";
- its Result. v2 passes on silicon (board 2, b42c8c644). The forced-failure control stops with
  `SUBLET-RENEW-FAILED` and rc 134, and is graded MISS, as predicted.

The port's own scripts are in `capstone/ports/spec-cpu2017/own-allocators/502.gcc_r/`. v2 differs from v1 in one
file.

- **`sublet-port.h`, the v2 header.** Its `sublet_renew` handles the untagged 0 that CRENEW returns when the table
  can give the renewal no new identity and cannot grow; in that case the old lifetime stays live. The header ends the
  old lifetime with CREVOKE, writes `SUBLET-RENEW-FAILED` to stderr, and aborts.
  - Under `SUBLET_RENEW_FORCE_FAIL`, every renewal reads as failed without executing CRENEW. That is the control
    build.
  - The rest of the header is v1's.
- **`build-gcc-v2.sh`** builds gcc-port2 and gcc-port2ff the way `build-ports.sh` builds gcc-port:
  - B5TL1's own 502.gcc_r tree, with its `compile.txt` and `link.txt`;
  - llvm-c9 and the B5TL1 SDK;
  - the 11 port sources, each first checked equal to the stock source.

  `inputs.txt` lists the compiler and the hashed inputs.
- **`build-bmfx.sh`** builds the port's bitmap fixture three ways: v1, v2 and v2ff.
- **`twin-gcc2.sh`** runs the twin requalification of the stages and fixtures.

The copies replace the account's paths with `<hdd>/` and `<home>/`, so they do not run as committed.
`RAN-SHA256.txt` gives the sha256 prefixes of the files as they ran. Those match the hashes the sections quote:
`9a132f3b` for `build-gcc-v2.sh`, `65348165` for `build-bmfx.sh`, and `d62bdcc5` for `twin-gcc2.sh`.
