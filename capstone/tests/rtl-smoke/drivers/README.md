# Board drivers, chains and invocation lists

The board lane's execution machinery, committed 2026-09-15 for the handover to another machine
(`docs/plans/2026-09-15-board-lane-handover.md`). Until then these lived in `~/capstone-artifacts/unify/`
and a session scratchpad with hard-coded paths; every path is now a knob with the old value as its
default, and the measurement logic is byte-for-byte what ran the boots the measurements doc cites.
Read the `board-run` skill first; it is the decision procedure these scripts implement.

## What runs a boot

| script | what it does | mandatory knobs |
|---|---|---|
| `board-r1e4.sh` | **the live entry point**: bake (three rebuilds under the memory lock) → stage the R1 harness image and the readback host → control (`lpc\|k800`) → the boot's 12 invocations from the list → control → summary through `fpga_driver.transcript` → marker (`done` / `failed` / `refused`) → restore-bake in the exit trap | `R1_BOOT` (1-based), `R1_IMG`, `R1_HASH` (sha256/16), `R1_LIST`, `R1_QEMU_LOG`, `R1_HOST` (the readback host) |
| `chain-r1.sh` | N boots of `board-r1e4.sh` over one list, ends with `CHAIN_<TAG>_DONE` | `CHAIN_TAG`, `CHAIN_BOOTS`, the `R1_*` set, `R1_BOOT_TAG`, `R1_BOOT_DESC`, `R1_PREREG` |
| `chain-m1.sh` | F6: the no-reclamation baseline, two boots, prepared and NOT launched | `M1_IMG`, `M1_HASH`, `M1_QEMU_LOG` |
| `qemu-m1-flowcheck.sh` | the emulator flow check of the M1 series (writes the qemu-pass record) | `R1_DOM`, `R1_HOST` |
| `board-c6var.sh` | F1's boot shape: control, a cell-6 variant image at the 2 MiB arena, control, probe | `C6_TAG`, `C6_IMG`, `C6_HASH`, `C6_QEMU_DEFAULT`, `C6_QEMU_DEFAULT_LOG`, `C6_QEMU_2MIB`, `C6_QEMU_2MIB_LOG`, `R1_HOST` |
| `board-b80s.sh`, `board-b80a.sh`, `board-b80b.sh`, `board-b79.sh`, `board-b78-w2h.sh` | the P1 -O2 cells, the `--stats` boot, E2's boot and E1's per-arm boot — **references for the record shape**; their images live in a scratchpad that is gone, so they fail loudly on the named knob (`CELL_IMG`, `NATIVE_BASELINE`, `QEMU_LOG_*`, `E1_DIR`) rather than on a dead path | see each header |

Optional knobs, every driver: `CAPSTONE_ARTIFACTS` (default `~/capstone-artifacts`; results land in
`unify/board-<tag>/`, pass records in `qemu-pass/`), `CAPSTONE_FPGA_URL_FILE` (default
`~/.claude-kisp/secrets/fpga-console-url`; never echoed, never committed), `MEMLOCK` (default
`~/bin/logs/machine-memory.lock`; a bake waits on it), `CAPSTONE_BR_FW`, `CAPSTONE_BR_TARGET`,
`CAPSTONE_BR_OVERLAY`, `CAPSTONE_BR_BUILD`, `CAPSTONE_KO` (the buildroot per-target layout: defaults are
`build-fpga/...` and `overlay/test-domains`, which is what both the original host and the apollo server have),
`ENTRY_STALL_S` (420; the JTAG upload is up to 227 s of legitimate silence), `WD_IDLE` (900),
`R1_HOST_HASH` (default `2c9e82d101b48160`, the host these drivers were run with).

## Gates the drivers keep (do not weaken any of them)

* the monitor at `4274268` and the FPGA buildroot copy at `d04bd83`, checked by commit;
* the image hash equals the knob, and `${CAPSTONE_ARTIFACTS}/qemu-pass/<sha256>` exists (the emulator
  pass, written by `sublet/r1/run-r1-qemu.sh` on `R1_RC=0`), and the emulator log holds the gate line;
* the readback host's hash, and its `RR/share` probe string; `lpc` on the overlay at `3b93a2b6e2adfa36`;
* the baked `.ko` carries the `#3` marker; the initramfs membership check; `preflight-board-run.sh`;
* no board runner live (`pgrep -f 'python3 -m fpga_driver'` inside the driver: the overlap guard);
* the marker agrees with the exit status (ISSUES M-11): `refused` on a preflight BLOCK, `failed` on
  any other non-zero, `done` only on rc 0 and a parsed transcript.

## Invocation lists (`lists/`)

One line per harness invocation: `run arm series pattern arena extra...` — `run` the repetition number,
`arm` S/P/D (or the M1 pattern name), `series`, `pattern`, the arena in bytes, then the harness's own
arguments. `board-r1e4.sh` takes list lines `12*(R1_BOOT-1)+1 .. 12*R1_BOOT`. `f5-chase.txt` is M2's
45-line seeded permutation (four boots); `m1-diag.txt` / `m1-capacity.txt` are F6's; `e4-calibration.txt`
and `f4-linear.txt` are E4's and F4's.

## Inputs the successor must produce (a rebuild is a new hash; cite the new one)

* the R1 harness image: `OUT_DIR=... R1_OPT=-O1 bash capstone/sublet/r1/build-r1-silicon.sh` (ends
  `VERDICT: fits`); then the emulator pass `R1_DOM=... R1_HOST=... OUT=... bash capstone/sublet/r1/run-r1-qemu.sh "<args>" <arena>`;
* the readback host `sqlite_host_rr.user`: `capstone/ports/sqlite/build-sqlite-host.sh` from
  `sqlite_host.c` with the module's `libcapstone.c`, its provenance recorded beside it (source, libcapstone
  sha, buildroot HEAD, dirty count); verify `strings | grep -c 'RR/share'` ≥ 1;
* the control rungs `lpc` and `k800.dom`: `capstone/tests/runtime-qemu/silicon-ladder/build-ladder-domain.sh`,
  then the verify step that writes the preflight's oracle and `.qemu-pass` under `PREFLIGHT_ORACLES`
  (default `/tmp/capstone/ladder-fpga`) — regenerate, never copy;
* the SQLite cells for F1: `capstone/ports/sqlite/build-sqlite-silicon.sh`.

## Reading a run

`unify/board-<tag>/`: `driver.log` (the framed runner log — the record; read it ONLY through
`fpga_driver.transcript`, ISSUES M-11), `boot.txt` (the run-scoped UART window: the tracked raw record for a
bundle), `watchdog.log`, `fw.sha`, `marker`, `r1-lines.txt`, `log`, the six bake logs, `retired/`. The
runner and watchdog logs are never committed (build-host identifiers in banners); bundles carry
`*-boot.txt` and result lines. Cite every result by the image hash from the run-scoped transcript.
