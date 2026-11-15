# tshark on the release's own test captures: reuse distance at wmem, pre-registration, 2026-11-15

Committed before any measured run of this workload. A reading that differs from what is written
here is a finding; this file is not edited afterwards. Definition, observer, hooks, arms and report
format are those of `2026-10-11-nested-reuse-distance.md` (16d2171a3edc). Only the input set and the
repetitions change.

## The question

Does the wmem lifetime patch change when tshark's wmem allocators reuse an address, on an
established input set instead of the port's ten captures?

## Where the current ten captures come from

`ports/wireshark/app/results/2026-10-11-reuse-distance/run-reuse.py` stages six captures of the
Wireshark 4.6.8 source tree's `test/captures/` (arp, dhcp, dns_port, http, ntp, dns-ooo) and four
copies of the first four with one byte flipped eight bytes from the end (the same-answers oracle's
changed-input control, `host/oracle.sh`). The `--captures` directory of the run-3 plan is the
build's `xsrc/test/captures`, which is byte-identical (`diff -r`) to `test/captures/` of the release
tarball. So the established input set is already the source of the ten; this plan extends them to
the whole directory and drops the flipped copies, which are not part of the release.

## Boundary

Unchanged: wmem `block` and `block_fast` object hand-out and free (`wmem-reuse-gap.patch`
b192b4c1..., `wmem-reuse-gap.h` 91f1e94c..., `reuse-gap-observer.h` 627b0fe3...), one report per
process and boundary, 65,536-slot table per boundary.

## Workload

- **Program:** Wireshark 4.6.8, https://www.wireshark.org/download/src/all-versions/wireshark-4.6.8.tar.xz,
  sha256 `c0f1ccf217bc0d3b51a9c03ea178b0f7df682e475da26a2d21cd4a1bdd9579d0`, built by
  `experiments/study/reuse-hooks/wireshark/build-wireshark-reuse.sh stock|protected` from trap/eval
  ed8cc375c2 (libc and lifetime patches of llvm-capstone native-kernel-new-qemu 7ba3f427f9).
- **Command:** `tshark -r C -V -n`, `TZ=UTC`, empty `HOME` and `WIRESHARK_CONFIG_DIR`, one process
  per capture, all captures of one arm and repetition in one boot, in the order of the set file.
- **Inputs:** the files of the tarball's `test/captures/` (202 files with the `mysql/` and `zabbix/`
  subdirectories). Candidates are every file except the 25 that are not captures: the 15 `*.cbordiag`
  texts, the two `opcua-*-keys.txt` key logs, `text2pcap_hash_eol.txt` (text2pcap input) and the
  seven zabbix `*.output` expectation files. That leaves 177 candidates (`*.pcap`, `*.pcapng`, their
  `.gz` forms, `many_interfaces.pcapng.1..3`, `canlogger-cl2000.txt`, `sipmsg.log`).
- **The measured set (169 captures):** the 177 candidates minus two rules, fixed by one stock
  calibration boot per quarter of the candidates on 2026-11-15 (run-3 stock image, the same kit,
  seven QEMU instances on the host at the time):
  - the six captures larger than 500,000 bytes uncompressed (grpc_web.pcapng.gz,
    challenge01_ooo_stream.pcapng.gz, netperfmeter.pcapng.gz, quic_follow_multistream.pcapng,
    wpa-test-decode.pcap.gz, opcua-signed.pcapng): they took 1,019 of the 3,633 guest seconds that
    all 177 took, and without them one boot fits the hour (2,614 s calibrated);
  - the two captures on which tshark itself exits nonzero (trunc.pcap, exit 14, a capture cut short
    in the middle of a packet; sipmsg.log, exit 3), which the acceptance rule below would reject.

  All 169 others exited 0 with REUSE-GAP lines at both boundaries (empty.pcap, which has no
  packet, only at wmem-block) and observer error 0. Together they handed out 153,761 wmem-block
  and 1,907,699 wmem-block-fast objects in 2,582 guest seconds; the largest per-process
  distinct-start count was 6,906 of the 65,536 slots. (The 13 captures in `mysql/` and `zabbix/`
  were rerun in a fifth calibration boot after a fix to the runner's markers, below.) The list,
  in the order run, is
  `experiments/study/reuse-hooks/wireshark/test-captures-4.6.8.txt` (sha256
  `c8d1114f412a16429ec1de636bc33834886d7bb9471802b6c3aeb61fc5864cff`); it contains the six
  captures the ten came from, but not their flipped copies.
- **Runner:** `run-reuse.py --set capstone/experiments/study/reuse-hooks/wireshark/test-captures-4.6.8.txt --rep N`
  (the October runner; `--set` replaces its capture list, records per-capture hashes and seconds,
  and writes a `rep.json` for `collect-reuse.py` with each boundary summed over the processes; a
  reuse distance never crosses a process). The guest markers' sed now uses `|` as delimiter: with
  `/` it dropped the REUSE-GAP lines of the captures in subdirectories.

## Platform

The run-3 trap-route kit `~/asr-work/eval/kit-run-3` on p13: QEMU capstone-qemu address-space-root
9289114e06 (`qemu-1b`), kernel `Image-0133b9e21cd`, firmware `f-keep/fw_jump.elf`, run-3 initramfs
and capstone-vexec. At most 8 QEMU instances at a time on the host.

## Arms and repetitions

stock and protected, two repetitions each (stock-1, stock-2, protected-1, protected-2), one boot
per arm and repetition. Results in `~/asr-work/eval/run-5/reuse/wireshark-<arm>-<rep>/`. Images
built on p13 for this plan: stock `tshark.dom` sha256 `971113450dc15a414c739888e0b32e007fd3e434b246408c54065a3277e1f2ed`,
protected `f77d31b60ac2ef987bda88fb53aaf2d1889d6df53c7916e299bbcd125dfb146a` (allocator sources
identical to the run-3 images'; the images differ from run-3's in build paths).

## Acceptance

A repetition is rejected if its boot did not complete, if any capture's process exits nonzero,
faults or lacks a REUSE-GAP line, if any observer `error` is nonzero, or if any capture's stdout
sha256 differs between the arms.

## Predictions

1. **Noise floor zero:** tshark is deterministic on a file. stock-1 and stock-2 give identical
   counters and bins at both boundaries, summed and per capture; so do protected-1 and protected-2
   (reusediff 0.0 within each arm).
2. **Equal arms:** stock and protected give the same histogram bucket for bucket at both
   boundaries, summed and per capture (reusediff 0.0), and equal issues, releases, reuses and
   distinct starts, as on the ten captures of 2026-10-11.
3. **Same output:** every capture's stdout hash is equal across all four repetitions.

The report gives, per arm and repetition, issues, releases, reuses, distinct starts and error per
boundary (sums over the processes), the output check, and the three reusediff values of
`collect-reuse.py` (stock-1 vs stock-2, protected-1 vs protected-2, stock vs protected).
