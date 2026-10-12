# The published PoisonCap platform

This directory used to hold the experimental FFmpeg pool integration with the
published PoisonCap platform. That integration -- its backend, build and runner --
was removed on 2026-10-10. What remains is the platform recipe, because live
work still boots it: the stock CheriBSD runners of the httpd and memcached
corpora use this platform's image with the libc fix in
`bug-corpora/cpython/pymalloc-repros/platform/`, and the PostgreSQL
memory-context port's PoisonCap arm builds on it. Recorded FFmpeg PoisonCap
results stay as history.

## Reconstruct the platform

`platform.json` pins the paper artifact, complete upstream source bases and
the missing compressed-capability dependency. `prepare.py` overlays published
files onto those bases, preserving omitted upstream dependencies. It also
installs the artifact's separately supplied version-aware capability header
into the dependency directory. Python bytecode caches are excluded.

Fetched sources, SDKs, images, reports and logs stay outside this repository:

```sh
source capstone/tests/capstone-test-env.sh
PORT=capstone/ports/ffmpeg/buffer-pool
PC="$PORT/host/cheribsd/poisoncap"
WORK=/tmp/capstone/poisoncap-work
python3 "$PC/prepare.py" "$WORK"
python3 "$PC/prepare.py" "$WORK" --verify

# Host dependencies include Clang/LLD 18, CMake, Ninja and CheriBSD build tools.
# Use the cheribuild revision recorded in platform.json.
export CHERIBUILD=/path/to/cheribuild/cheribuild.py
export CHERI_BOOT_FIRMWARE=/path/to/sdk/share/qemu/bbl-riscv64cheri-virt-fw_jump.bin
bash "$PC/platform.sh" "$WORK" llvm
bash "$PC/platform.sh" "$WORK" qemu
bash "$PC/platform.sh" "$WORK" cheribsd
bash "$PC/platform.sh" "$WORK" image
```

The CheriBSD build keeps Kerberos enabled: this revision's libc includes a
GSSAPI header even in an otherwise minimal build. The port enables GNU C for
the published revocation header and explicitly selects LLD when linking.

Existing artifact archives can be supplied as `--artifact` and `--cap-library`.
The build stages do not alter existing standard CheriBSD installations.
The older emulator executable is named `qemu-system-riscv64xcheri`; the SDK
provides the name expected by the shared runner. Firmware is an explicit
external input and must be included in the run's platform fingerprints.
