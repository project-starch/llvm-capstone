# wireshark/wmem-repros on PoisonCap, cases 13-21, each with its own label's supervisor -- 2026-10-10 (R4)

Pre-registered at `49e8e23f4122` (R4). The fixed runner (`ports/wireshark/wmem/host/cheribsd/poisoncap/run.py`)
resolves each case's own label from its case.c; a write case runs under supervise-wm_defect_write. As predicted,
every arm faults (SIGPROT, bounds) AT its case's label in BOTH modes -- what the 2026-10-09 run needed a hand
attribution for (cases 16, 17, 18, 20, 21) the runner now reports itself.

    python3 ports/wireshark/wmem/host/cheribsd/poisoncap/run.py <build> <out> --modes 0,1 --runtime-revocation on \
      --sdk ... --rootfs ... --image ... --case 13-...-mode0 ... (cases 13-21)

Platform (the PoisonCap image): qemu 5a9a8ef9cade9d92, firmware f0e1fe57b0f85075, kernel 7b4b2b5873f08c69, libc 8071c62d364f388f, image c8df9e17594b2614.

| arm | exit | label | fault pc == label |
|---|---|---|---|
| 13-http-range-cursor-past-chunk-mode0 | 162 | wm_defect_probe | yes |
| 13-http-range-cursor-past-chunk-mode1 | 162 | wm_defect_probe | yes |
| 14-solaredge-payload-six-past-mode0 | 162 | wm_defect_probe | yes |
| 14-solaredge-payload-six-past-mode1 | 162 | wm_defect_probe | yes |
| 15-opcua-padding-below-chunk-mode0 | 162 | wm_defect_probe | yes |
| 15-opcua-padding-below-chunk-mode1 | 162 | wm_defect_probe | yes |
| 16-dcp-etsi-rs-parity-write-mode0 | 162 | wm_defect_write | yes |
| 16-dcp-etsi-rs-parity-write-mode1 | 162 | wm_defect_write | yes |
| 17-dns-one-byte-write-mode0 | 162 | wm_defect_write | yes |
| 17-dns-one-byte-write-mode1 | 162 | wm_defect_write | yes |
| 18-rtps-batch-sample-info-unguarded-mode0 | 162 | wm_defect_write | yes |
| 18-rtps-batch-sample-info-unguarded-mode1 | 162 | wm_defect_write | yes |
| 19-ntlmssp-blob-length-before-check-mode0 | 162 | wm_defect_probe | yes |
| 19-ntlmssp-blob-length-before-check-mode1 | 162 | wm_defect_probe | yes |
| 20-proto-undecoded-bitmap-unbounded-mode0 | 162 | wm_defect_write | yes |
| 20-proto-undecoded-bitmap-unbounded-mode1 | 162 | wm_defect_write | yes |
| 21-tcp-flags-str-sixteen-bytes-mode0 | 162 | wm_defect_write | yes |
| 21-tcp-flags-str-sixteen-bytes-mode1 | 162 | wm_defect_write | yes |
