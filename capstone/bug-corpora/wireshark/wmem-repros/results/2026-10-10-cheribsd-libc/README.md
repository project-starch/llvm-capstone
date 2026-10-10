# wireshark/wmem-repros on stock CheriBSD, wmem on libc (WM_LIBC_SYSTEM) -- 2026-10-10

**What this replaces, and what it shows.** `../20261007-cheribsd-revocation-control/` measured the CHERI column on
the hosted bump arena with the chunk port compiled in (`WM_CHUNKS=ON`), where no wmem object could reach libc
`free()` and "completes" was the only possible reading. This run builds STOCK wmem (`WM_CHUNKS=OFF`) with its system
allocator as libc (`WM_LIBC_SYSTEM=ON`: `g_malloc`/`g_free` are CheriBSD's `malloc`/`free`, libc revocation on), and
runs every case in the fix-differential invocation, where a temporal case ASSERTS that the next dissection's first
allocation reoccupies the freed storage before reading through the stale pointer -- `wm_reoccupy`'s
`CHECK(next == stale)` in cases 02-10, and the same check written inline (`CHECK(next == address)`) in 00, 01, 11
and 12. A failed assertion exits 75, CONTROL-FAILED (`wm_give_up`: "an infrastructure failure is never a verdict"):
it would have meant the chunk was not reissued, as a quarantine hold would do, but the run would have reported it as
no reading rather than scored it held. None failed.

- **All 22 buggy arms: VERDICT DEFECT-REPRODUCED, exit 0.** For the 13 temporal cases (00-12) that is the
  reoccupation assertion holding: the stale chunk was REISSUED inside the block wmem kept, so the quarantine never
  saw it -- CHERI reads "missed (reused)", not "held". The 9 spatial cases (13-21) cross inside one `g_malloc`'d
  block. All 22 fixed arms: VERDICT FIXED.
- **Control 90** -- a jumbo `wmem_free_all` hands to `g_free`, then read -- read its own fill (`WM_READ(held, 0x5a)`
  true, nothing reoccupied it): the chunk was HELD in libc's quarantine. That is CheriBSD's mechanism acting on
  wmem's own free, the one sequence where a stock wmem object does reach `free()`.
- **revocation-control** faulted at its labelled load (SIGPROT si_code 2, addr = pc = the resolved probe): the revoker
  sweeps in this guest. The platform controls passed (`cheribsd-bounds` exit 162).

As pre-registered (`docs/history/10-10-2026_17-20-00_bug-corpus-audit-three-programs.md`, P1, commit `bbc7db0bd3d5`): every
prediction held. Image `0cb16209…`, the vehicle of every other CheriBSD reading in the three programs.

Pre-registration: bbc7db0bd3d5 (P1). Runtime revocation: on (guest default preserved).
Platform sha256: qemu 16135483052dfdd6, firmware f0e1fe57b0f85075, kernel 8ab453f46dc76cf2, libc fdce2289224bb519, image 0cb16209c16c5edf

Result LINES only; the boot capture is contaminated by construction.

## Programs: 48 of 48 passed their oracle

    PASS cheribsd-abi  exit=0 (expected 0)  program e403dbb33335dfe0
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
    PASS cheribsd-bounds  exit=162 (expected 162)  program e403dbb33335dfe0
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
        CHERI_BOUNDARY_READY
    PASS revocation-control  exit=162 (expected 162)  program 0c680eb0641bd5a8 target f2b99b0a4bad1cfa
        SUPERVISE expect mc_defect_read 0x101dc0
        REVOCATION_CONTROL revocation=1 tag_after_sweep=0 reissued=1
        SUPERVISE fault signal=34 code=2 addr=0x101dc0 pc=0x101dc0
        SUPERVISE exit signalled=34
    PASS 00-rpcrdma-write-offsets-global-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 87408ada7ecfd29c
        SUPERVISE expect wm_defect_probe 0x103f68
        WM_DEFECT case=0 ready
        WM_DEFECT case=0 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 00-rpcrdma-write-offsets-global-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 87408ada7ecfd29c
        SUPERVISE expect wm_defect_probe 0x103f68
        WM_DEFECT case=0 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 01-cms-oid-global-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 21e0460bc77dabeb
        SUPERVISE expect wm_defect_probe 0x103fb4
        WM_DEFECT case=1 ready
        WM_DEFECT case=1 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 01-cms-oid-global-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 21e0460bc77dabeb
        SUPERVISE expect wm_defect_probe 0x103fb4
        WM_DEFECT case=1 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 02-mdb-address-column-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target baf7896d5be99c25
        SUPERVISE expect wm_defect_probe 0x103f04
        WM_DEFECT case=2 ready
        WM_DEFECT case=2 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 02-mdb-address-column-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target baf7896d5be99c25
        SUPERVISE expect wm_defect_probe 0x103f04
        WM_DEFECT case=2 ready
        WM_DEFECT case=2 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 03-cola-info-column-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 943cd33bd785ed06
        SUPERVISE expect wm_defect_probe 0x103f04
        WM_DEFECT case=3 ready
        WM_DEFECT case=3 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 03-cola-info-column-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 943cd33bd785ed06
        SUPERVISE expect wm_defect_probe 0x103f04
        WM_DEFECT case=3 ready
        WM_DEFECT case=3 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 04-qnet6-col-set-str-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 5ba28ac5751fa13e
        SUPERVISE expect wm_defect_probe 0x103f58
        WM_DEFECT case=4 ready
        WM_DEFECT case=4 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 04-qnet6-col-set-str-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 5ba28ac5751fa13e
        SUPERVISE expect wm_defect_probe 0x103f58
        WM_DEFECT case=4 ready
        WM_DEFECT case=4 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 05-usbll-address-struct-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 438e972969370cb4
        SUPERVISE expect wm_defect_probe 0x103eba
        WM_DEFECT case=5 ready
        WM_DEFECT case=5 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 05-usbll-address-struct-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 438e972969370cb4
        SUPERVISE expect wm_defect_probe 0x103eba
        WM_DEFECT case=5 ready
        WM_DEFECT case=5 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 06-x509if-last-dn-static-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target b8410c65a6ebf7b4
        SUPERVISE expect wm_defect_probe 0x103f14
        WM_DEFECT case=6 ready
        WM_DEFECT case=6 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 06-x509if-last-dn-static-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target b8410c65a6ebf7b4
        SUPERVISE expect wm_defect_probe 0x103f14
        WM_DEFECT case=6 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 07-mysql-auth-method-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 85be3369a07cfd8b
        SUPERVISE expect wm_defect_probe 0x103f54
        WM_DEFECT case=7 ready
        WM_DEFECT case=7 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 07-mysql-auth-method-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 85be3369a07cfd8b
        SUPERVISE expect wm_defect_probe 0x103f54
        WM_DEFECT case=7 ready
        WM_DEFECT case=7 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 08-sip-cseq-method-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 3b80691b7bd6c9be
        SUPERVISE expect wm_defect_probe 0x103f8c
        WM_DEFECT case=8 ready
        WM_DEFECT case=8 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 08-sip-cseq-method-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 3b80691b7bd6c9be
        SUPERVISE expect wm_defect_probe 0x103f8c
        WM_DEFECT case=8 ready
        WM_DEFECT case=8 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 09-geonw-proto-data-tvb-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 306ad36a52ffe623
        SUPERVISE expect wm_defect_probe 0x103f4c
        WM_DEFECT case=9 ready
        WM_DEFECT case=9 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 09-geonw-proto-data-tvb-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 306ad36a52ffe623
        SUPERVISE expect wm_defect_probe 0x103f4c
        WM_DEFECT case=9 ready
        WM_DEFECT case=9 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 10-t38-reassembly-buffer-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 3c55dbdca022872a
        SUPERVISE expect wm_defect_probe 0x103f00
        WM_DEFECT case=10 ready
        WM_DEFECT case=10 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 10-t38-reassembly-buffer-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 3c55dbdca022872a
        SUPERVISE expect wm_defect_probe 0x103f00
        WM_DEFECT case=10 ready
        WM_DEFECT case=10 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 11-http-header-map-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 52ee11ea7bd1e303
        SUPERVISE expect wm_defect_probe 0x103f74
        WM_DEFECT case=11 ready
        WM_DEFECT case=11 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 11-http-header-map-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 52ee11ea7bd1e303
        SUPERVISE expect wm_defect_probe 0x103f74
        WM_DEFECT case=11 ready
        WM_DEFECT case=11 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 12-xml-root-name-recycled-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 50b460c5d862b0fb
        SUPERVISE expect wm_defect_probe 0x103fa0
        WM_DEFECT case=12 ready
        WM_DEFECT case=12 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 12-xml-root-name-recycled-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 50b460c5d862b0fb
        SUPERVISE expect wm_defect_probe 0x103fa0
        WM_DEFECT case=12 ready
        WM_DEFECT case=12 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 13-http-range-cursor-past-chunk-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target a21a410e81a7ad47
        SUPERVISE expect wm_defect_probe 0x1040a0
        WM_DEFECT case=13 ready
        WM_DEFECT case=13 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 13-http-range-cursor-past-chunk-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target a21a410e81a7ad47
        SUPERVISE expect wm_defect_probe 0x1040a0
        WM_DEFECT case=13 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 14-solaredge-payload-six-past-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 8439b752ffbbd814
        SUPERVISE expect wm_defect_probe 0x10400c
        WM_DEFECT case=14 ready
        WM_DEFECT case=14 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 14-solaredge-payload-six-past-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 8439b752ffbbd814
        SUPERVISE expect wm_defect_probe 0x10400c
        WM_DEFECT case=14 ready
        WM_DEFECT case=14 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 15-opcua-padding-below-chunk-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 1b853b90cc43b331
        SUPERVISE expect wm_defect_probe 0x103fa2
        WM_DEFECT case=15 ready
        WM_DEFECT case=15 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 15-opcua-padding-below-chunk-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 1b853b90cc43b331
        SUPERVISE expect wm_defect_probe 0x103fa2
        WM_DEFECT case=15 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 16-dcp-etsi-rs-parity-write-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 14f8ae3ceac7d169
        SUPERVISE expect wm_defect_probe 0x103fba
        WM_DEFECT case=16 ready
        WM_DEFECT case=16 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 16-dcp-etsi-rs-parity-write-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 14f8ae3ceac7d169
        SUPERVISE expect wm_defect_probe 0x103fba
        WM_DEFECT case=16 ready
        WM_DEFECT case=16 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 17-dns-one-byte-write-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 6c7df2a2ee1f5340
        SUPERVISE expect wm_defect_probe 0x103f7c
        WM_DEFECT case=17 ready
        WM_DEFECT case=17 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 17-dns-one-byte-write-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 6c7df2a2ee1f5340
        SUPERVISE expect wm_defect_probe 0x103f7c
        WM_DEFECT case=17 ready
        WM_DEFECT case=17 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 18-rtps-batch-sample-info-unguarded-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target e36603e4180456e6
        SUPERVISE expect wm_defect_probe 0x104024
        WM_DEFECT case=18 ready
        WM_DEFECT case=18 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 18-rtps-batch-sample-info-unguarded-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target e36603e4180456e6
        SUPERVISE expect wm_defect_probe 0x104024
        WM_DEFECT case=18 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 19-ntlmssp-blob-length-before-check-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 58a54bf465525e6f
        SUPERVISE expect wm_defect_probe 0x104008
        WM_DEFECT case=19 ready
        WM_DEFECT case=19 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 19-ntlmssp-blob-length-before-check-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target 58a54bf465525e6f
        SUPERVISE expect wm_defect_probe 0x104008
        WM_DEFECT case=19 ready
        WM_DEFECT case=19 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 20-proto-undecoded-bitmap-unbounded-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target b4ec3d93c2664f1b
        SUPERVISE expect wm_defect_probe 0x104012
        WM_DEFECT case=20 ready
        WM_DEFECT case=20 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 20-proto-undecoded-bitmap-unbounded-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target b4ec3d93c2664f1b
        SUPERVISE expect wm_defect_probe 0x104012
        WM_DEFECT case=20 ready
        WM_DEFECT case=20 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 21-tcp-flags-str-sixteen-bytes-buggy  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target ac5f9dd17cb93f95
        SUPERVISE expect wm_defect_probe 0x103f86
        WM_DEFECT case=21 ready
        WM_DEFECT case=21 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
    PASS 21-tcp-flags-str-sixteen-bytes-fixed  exit=0 (expected 0)  program 176a1d2a0b1ef0f2 target ac5f9dd17cb93f95
        SUPERVISE expect wm_defect_probe 0x103f86
        WM_DEFECT case=21 ready
        WM_DEFECT case=21 mode=0 completed
        VERDICT FIXED the fix's sequence does not reach another object's storage
        SUPERVISE exit status=0
    PASS 90-jumbo-freed-by-reset-buggy  exit=0 (expected 0)  program b34775e6bf24c981 target b5ee35499d289fb9
        SUPERVISE expect wm_defect_probe 0x103e74
        WM_DEFECT case=90 ready
        WM_DEFECT case=90 mode=0 completed
        VERDICT DEFECT-REPRODUCED the access reached storage outside its object
        SUPERVISE exit status=0
