# Patch 0011 twin campaign: pre-registration (2026-11-13, before any arm ran)

Arms, one variable: A = nat-fair-cache SDK eb73a2bf475c (perf/nat-fair-cache/stages), B = this branch f097dd782aae
(perf/natrealloc/stages, regenerated from the same source stages; run.sh identical up to the directory name). Both on
the native twin, kernel perf/natio/images (native-io-align k3), 936M, direct arm, --no-sampler, capmix3 shift 24
around each command (prof-stage.sh), vmstat page counters and native_counters at each command's start and end.

Predictions (B against A):
- 505.mcf_r test: pages allocated (pgalloc_dma32 delta) A ~394k -> B 200k-270k; kernel instructions in the command
  windows fall 0.8-1.3 G; user instructions fall 20-150 M (no copy of the 236 MB arrays); outputs pass specdiff.
- 502.gcc_r train01, 538.imagick_r test, 557.xz_r test, 500.perlbench_r test, 525.x264_r test: outputs identical to
  A's (specdiff / host validation); kernel instructions within +-5 %.
- Failure branch: mcf pages above 300k means the in-place path did not take the arc arrays; then read why before any
  other claim.

## Arm C: the mapping cache makes room by eviction (written before any C run)

C = B + malloc-mapping.c's mr_make_room (perf/natevict/stages), same twin configuration. Predictions (C against B):
- 505.mcf_r test: 7 of the 8 arcs_pointer_sorted mappings (64,094,208 B) reused: pages fall 90k-125k; kernel
  instructions fall 0.5-0.8 G; user instructions rise 30-150 M (the reuse path's memset of 61 MB, 7 times).
- 502.gcc_r train01: service_other within +-15 % of B's (evictions must not cost the small-mapping hits); outputs
  identical to A's. The other benchmarks: outputs identical, kernel within +-5 %.
- mapping-reuse-probe on the native twin, every mode, A/B/C: the same verdicts on all three; C's pressure count equal
  to A's. Failure branch: gcc service_other more than 15 % above B's means eviction thrashes the small classes.
