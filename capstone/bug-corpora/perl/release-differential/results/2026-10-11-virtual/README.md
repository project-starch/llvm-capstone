# Perl 5.36.3 on the Sublet platform and on CheriBSD, 2026-10-11

What was predicted, and each change made after a run, is in [PREDICTIONS.md](PREDICTIONS.md) and
[../2026-10-11-cheribsd/PREDICTIONS.md](../2026-10-11-cheribsd/PREDICTIONS.md), each committed
before the run it governs (ced629f24256, amended in 2733d84ca12f). The first run is kept under
`first-run/`; the rows below are the rerun on images rebuilt with port patch 0010.

## The arms

| arm | what | image |
|---|---|---|
| `virtual-malloc` | Perl as released on the virtual profile's musl mallocng, which bounds each object and retires its lifetime on free | 296cb582fd87 |
| `virtual-nested-pools` | the same with `PERLD_SUBLET=1`: every SV head a CDERIVE child of its arena, revoked by CREVOKE when freed (`ports/perl/sublet/patches/5.36.3/0001`) | 317ae110b69f |
| `cheribsd-revocation` | purecap Perl (`ports/perl/cheribsd/build.sh`, dynamic), stock image, revocation on system-wide; each case also with revocation off for the process | 0f2aef270453 |

Both images carry port patches 0001-0010. Platform: the Sublet-lifetime QEMU af37cc32, module
a1c6cb6b, launcher 3975314a (`inputs.json`). CheriBSD: image e7470361, 15.0-CURRENT,
`runtime_revocation_default=1`, helpers sicode.so 511a3074, quarantine-probe.so 4f68353c
(`../2026-10-11-cheribsd/run/`). Every control passed on every arm.

## Rows

| case | layer | virtual-malloc | virtual-nested-pools | CheriBSD, revocation on (off) |
|---|---|---|---|---|
| 01 | nested | fault 24, Perl_pp_iter | fault 25, Perl_pp_iter | tag fault, Perl_pp_iter (the same off) |
| 02 | nested | exit 255, Perl's own "attempt to copy freed scalar" | fault 25, Perl_sv_setsv_flags | Perl's panic (the same off) |
| 03 | system | completed, [PASS] | the same | completed (the same off) |
| 04 | system | **fault 25, memmove (site)** | the same | completed; 4,936 frees, all quarantined, none reissued (off: completed) |
| 05 | nested | completed, [FAIL] | **fault 25, Perl_cv_undef_flags** | completed (the same off) |
| 06 | system | completed, [FAIL] 5 of 29 | the same | completed; 8,402 frees, all quarantined, none reissued (off: tag fault in Perl_regfree_internal) |
| 07 | system | **fault 25, S_regcppop (site)** | the same | completed; 4,564 frees, all quarantined, none reissued (off: completed) |
| 08 | spatial | fault 28, Perl_pp_gvsv | the same | tag fault, Perl_sv_setsv_flags (the same off) |
| 09 | nested | fault 24, Perl_newATTRSUB_x | fault 25, Perl_newATTRSUB_x | tag fault, Perl_newATTRSUB_x (the same off) |
| 10 | system | **fault 25, memmove (site)** | the same | completed, [PASS]; 4,288 frees, all quarantined, none reissued (off: completed) |
| 11 | -- | completed, [FAIL] | the same | completed (the same off) |

## Read by the registered rules

| | cases | CheriBSD (use after reallocation) | Sublet (use after free) |
|---|---:|---:|---:|
| system allocator: 04, 07, 10 | 3 | 3 | 3 |
| nested allocator: 02, 05 | 2 | 0 | 2 |

* **System allocator.** 03 is out: its trigger does not reproduce here (nor on any arm since
  2026-10-06). **06 is out: it is not a temporal defect** (correction, 2026-10-11, after the first
  version of this file counted it). One count serves as both the array's size and its used
  entries, so `S_free_codeblocks` walks entries whose `src_regex` was never written. Host ASan
  reports a SEGV on a high-value address with its default fill of new memory and nothing at all
  with `malloc_fill_byte=0`, where the test gives the same five wrong results as on both platforms
  here ([case06-asan-fill.txt](case06-asan-fill.txt)); ASan flags every read of freed or
  out-of-bounds memory, so no ended lifetime is involved. It completing on both Capstone images is
  therefore not a miss, and CheriBSD's completion with revocation on is not use-after-reallocation
  protection (its revocation-off fault is a dereference of an uninitialized pointer, which tag
  integrity rejects). Sublet catches 04, 07 and 10 with cause 25 at a host ASan site. CheriBSD
  earns use-after-reallocation credit on all three: each completes with revocation on while every
  free stays in the quarantine and none is reissued.
* **Nested allocator.** A case counts when the stock image reaches it and ends without a fault.
  02 (Perl's own panic) and 05 qualify, and the protected image faults on both with cause 25: on
  02 at `sv_setsv_flags`' first read of the freed SV the defect hands it, one step before Perl's
  own check; on 05 in `Perl_cv_undef_flags`. 01 and 09 have no baseline: the stock image faults
  first with cause 24, where a reissued head's contents are followed; the protected image faults
  earlier there with cause 25 at the same site. CheriBSD earns nothing on 02 and 05: an SV head
  goes back on `PL_sv_root` and never reaches `free()`, so its quarantine never holds one (the
  2026-10-06 probe showed 05's head is not among its frees); 05 completes, 02 ends in Perl's panic.

## Caveats

* The stock virtual port faults on its own in several of Perl's core test files (`sv_clear` on
  tied-hash elements, the mro functions, `pp_aassign`, `pp_mapwhile`), identically on both images
  (`qualify-v2.txt`, 47 files, on the images before patch 0010). No corpus case hit one of them
  in the rerun.
* The protected image faults on one correct program the stock image runs: `$x = *foo; *x = $x`,
  where upstream reads a glob after `LEAVE` may have freed it.
* One run per case (N = 1).
