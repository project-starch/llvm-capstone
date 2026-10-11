# wmem-repros 22 (c702b44a01, USB HID double free) on every arm -- 2026-10-11 (R9)

Pre-registered at `206cc3c26384` (docs/history/10-10-2026_23-30-00_whole-corpus-audit-cross-program.md, R9). **11 of 11 arms as predicted**, and mode 0 on the chunk-port build stopped at the port's own double-free check, also as predicted (an observation, not a cell). Result LINES only; the captures stay out of the tree.

| arm | reading | predicted |
|---|---|---|
| native-fix-differential | buggy DEFECT-REPRODUCED (the INPUT field's freed array lost from the free list), fixed FIXED | two-sided |
| native-detect (ASan) | no report; both ASan controls reported in the same run | no report |
| cheribsd-revocation (stock, WM_LIBC_SYSTEM) | buggy DEFECT-REPRODUCED exit 0, fixed FIXED; the revocation control faulted at its label | complete |
| poisoncap-spatial (mode 0) | completes | complete |
| poisoncap-protected (mode 1) | SIGPROT si_code 2 at pc 0x104e6e = wm_widen (poisoncap.c:173), called from wmem_block_free (wmem_allocator_block.c:1272) -- a declared site | SIGPROT in wm_widen |
| spatial (region build, mode 0) | completes | complete |
| sublet (region build, mode 1) | completes; case 11 in the same session faulted, cause 24, at its read probe | complete |
| sublet-chunks (chunk port, mode 1) | cause 24 at the published allocator probe (wm_widen_probe), pc equal to the expected | fault at wm_widen_probe |
| sublet-malloc (reference build, mode 1) | completes; control 90 of the same options faulted, cause 24, at its read probe | complete |
| virtual-malloc | MISSED, DEFECT-REPRODUCED; controls bounds/uaf-malloc and 90 faulted, 91 completed | MISSED |
| virtual-nested-pools | CAUGHT, cause 24 at wm_widen_probe's own instruction (in static probe()); all four controls faulted | CAUGHT at wm_widen_probe |
| (observation) chunk port, mode 0 | `WM return=260`: wm_chunk_of's own `!chunk->used` check refused the second free | wm_fail(260) |

The chunk port catches the double free twice over: mode 1's revocation faults at the handback probe before the port's bookkeeping is consulted, and with nothing revoked (mode 0) that bookkeeping still refuses the second free. Every other arm misses it: the two frees never leave wmem's BLOCK allocator, so neither libc, ASan, virtual mallocng nor a region-granular Sublet sees an event. PoisonCap's per-chunk poisoning catches it at wm_widen.

## Result lines

native (`runners/run-native.sh`):

    22_c702b44a01_usbhid_output_usages_freed_twice buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED

ASan (`runners/run-asan.sh`):

      control 0: heap-use-after-free (required heap-use-after-free) ok  wm_probe <scratch>/w2/capstone/bug-corpora/wireshark/wmem-repros/controls/sublet-malloc/shared/driver.c:51
      control 1: heap-buffer-overflow (required heap-buffer-overflow) ok  touch <scratch>/w2/capstone/bug-corpora/tools/asan-control.c:14
      22_c702b44a01_usbhid_output_usages_freed_twice                 SILENT           the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing

Capstone domain, dom-region (`shared/run-defects.py`, Linux guest, capstone-qemu 6e5e136a070b):

    OK   case=11 11-http-header-map                       spatial
    OK   case=11 11-http-header-map                       sublet   cause=24 pc=0x1019015c8 expected=0x1019015c8 site=read
    OK   case=22 22-usbhid-output-usages-freed-twice      spatial
    OK   case=22 22-usbhid-output-usages-freed-twice      sublet   (recorded non-detection: completion required)
    4/4 arms passed

Capstone domain, dom-chunks (`shared/run-defects.py`, Linux guest, capstone-qemu 6e5e136a070b):

    OK   case=22 22-usbhid-output-usages-freed-twice      sublet   cause=24 pc=0x101901d74 expected=0x101901d74 site=allocator
    1/1 arms passed

Capstone domain, dom-ref (`shared/run-defects.py`, Linux guest, capstone-qemu 6e5e136a070b):

    OK   case=22 22-usbhid-output-usages-freed-twice      sublet   (recorded non-detection: completion required)
    1/1 arms passed

Capstone domain, dom-ctl90 (`shared/run-defects.py`, Linux guest, capstone-qemu 6e5e136a070b):

    OK   case=90 90-jumbo-freed-by-reset                  sublet   cause=24 pc=0x101901278 expected=0x101901278 site=read
    1/1 arms passed

Capstone domain, chunk port, mode 0 (observation):

    SESSION ENDED BEFORE WM_DEFECT_DONE, with no fault, for case=22 mode=spatial: /tmp/capstone/a2/r9/dom-chunks-mode0/22-usbhid-output-usages-freed-twice-spatial-aqfkofpj
    WM return=260 status=260 completed=0 allocs=0

stock CheriBSD (`ports/common/host/cheribsd/run.py`, revocation on):

    PASS cheribsd-abi exit=0 (expected 0)
    PASS cheribsd-bounds exit=162 (expected 162)
    PASS revocation-control exit=162 (expected 162)
    PASS 22-usbhid-output-usages-freed-twice-buggy exit=0 (expected 0)
    PASS 22-usbhid-output-usages-freed-twice-fixed exit=0 (expected 0)

    platform: qemu 16135483052dfdd6, firmware f0e1fe57b0f85075, kernel 8ab453f46dc76cf2, libc fdce2289224bb519, image 0cb16209c16c5edf

PoisonCap (`ports/wireshark/wmem/host/cheribsd/poisoncap/run.py`, modes 0 and 1):

    case 22: mode0 completed, mode1 faulted at a declared site: True

    SUPERVISE base 0x100000 /tmp/allocator-tests/22-usbhid-output-usages-freed-twice-mode1/22-usbhid-output-usages-freed-twice
    SUPERVISE expect wm_defect_probe 0x104250
    WM_POISONCAP mode=1 sweeps=2 poison_bytes=128 epochs=0 released_chunks=2 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
    SUPERVISE fault signal=34 code=2 addr=0x104e6e pc=0x104e6e
    SUPERVISE fault-object 0x104e6e in /tmp/allocator-tests/22-usbhid-output-usages-freed-twice-mode1/22-usbhid-output-usages-freed-twice base=0x100000
    SUPERVISE fault-ra 0x106c00 in /tmp/allocator-tests/22-usbhid-output-usages-freed-twice-mode1/22-usbhid-output-usages-freed-twice base=0x100000
    SUPERVISE exit signalled=34

virtual-malloc (`tools/run-virtual-cases.py --prebuilt`, --only 22):

      control bounds-malloc    fault     (expected fault)
      control uaf-malloc       fault     (expected fault)
      control jumbo-reset-90   fault     (expected fault)
      control chunk-free-91    complete  (expected complete)
    22_c702b44a01_usbhid_output_usages_freed_twice               MISSED  reached (the case printed `WM_DEFECT case=22 ready`, the last thing before its defective a
    --- virtual-malloc (virtual-wmem-libc): {'MISSED': 1}

virtual-nested-pools (`tools/run-virtual-cases.py --prebuilt`, --only 22):

      control bounds-malloc    fault     (expected fault)
      control uaf-malloc       fault     (expected fault)
      control jumbo-reset-90   fault     (expected fault)
      control chunk-free-91    fault     (expected fault)
    22_c702b44a01_usbhid_output_usages_freed_twice               CAUGHT  cause=24 pc=0x3f94b123bc in probe; by function: the fault is at wm_widen_probe, a site the
    --- virtual-nested-pools (virtual-wmem-chunks): {'CAUGHT': 1}

## Images (sha256/16)

- domain, region (WM_CHUNKS=OFF): `4d1f4efcb28edde3`
- domain, chunk port (WM_CHUNKS=ON): `4f1e5cc39cc377f7`
- domain, reference (WM_VARIANT=reference): `3a98fee4d8222467`
- domain, control 90 (reference): `c4f5f519b916bb92`
- CheriBSD libc (WM_LIBC_SYSTEM): `d3fd88c39bf34164`
- PoisonCap: `b2c0d308c1f00b14`
- virtual, libc (WM_LIBC_SYSTEM): `a2426cec83065676`
- virtual, Sublet chunks (WM_SUBLET): `083d354c35f8cb59`

