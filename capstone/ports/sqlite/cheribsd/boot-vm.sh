#!/bin/bash
# Boot the pinned CheriBSD riscv64-purecap image under the pinned QEMU, SSH on
# localhost:10086. NOTE: do NOT pipe `yes y` into this like the build script does --
# for the `run` target stdin IS the guest's serial console, so that floods the
# console with 'y' and wedges login. Dependencies (incl. the bbl/OpenSBI firmware
# bbl-riscv64cheri-virt-fw_jump.bin) are already built.
source /home/zephyr/cheriBSD/env.sh
# cb is a shell function from env.sh, so it cannot be exec'd.
cb run-riscv64-purecap --run/ssh-forwarding-port 10086 < /dev/null
