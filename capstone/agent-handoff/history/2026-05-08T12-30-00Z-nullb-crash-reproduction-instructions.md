# Full command sequence to reproduce the `null_blk` crash from scratch

This file is the direct answer to the question:

> "Write the full sequence of commands needed to reproduce this bug, for a person who does not have any of the repositories yet and will clone everything from scratch."

Below is one reproducible path for a clean Debian/Ubuntu machine where the person **does not yet have any of the required repositories**.

---

## What exactly we are reproducing

The goal is to reach the guest-side crash while running the official `null_blk` reference path:

```text
modprobe configfs
insmod /capstone.ko
/null_blk.user
insmod /nullb/capstone_split/null_blk.ko
```

Expected result: a kernel/module crash when running `insmod /nullb/capstone_split/null_blk.ko`.

---

## Repositories and pinned commits

Use exactly these repository URLs and commits:

- outer workspace repo:
  - `https://github.com/project-starch/llvm-capstone`
  - commit: `c0feaee7f78081ea58205b33b69540992e3823e2`
- Buildroot/runtime repo:
  - `https://github.com/project-starch/caplifive-buildroot.git`
  - commit: `3e6dbf74f1aa5ed03e4b867fca31f27605b14afc`
- QEMU repo:
  - `https://github.com/project-starch/capstone-qemu.git`
  - commit: `676f43c368297d6b6846d212f9a28a9860f44ede`
- Capstone-C compiler repo:
  - `https://github.com/jasonyu1996/capstone-c`
  - commit: `4899cf90dbd3a0d6b1b730a64dafca949e51a113`

---

## 0. Install host-side dependencies

The commands below are derived from the local build scripts (`local_build.sh`) and from the Python harness, which uses `pexpect`.

```bash
sudo apt-get update
sudo apt-get install -y \
  git cmake ninja-build build-essential \
  wget rsync curl bc \
  libglib2.0-dev libfdt-dev libpixman-1-dev libslirp-dev \
  libncurses5-dev libssl-dev zlib1g-dev \
  file cpio unzip sed make binutils diffutils patch perl tar findutils bzip2 \
  expect python3 python3-pip python3-venv python3-pexpect
```

---

## 1. Clone all repositories into the expected layout

This layout matters because `caplifive-buildroot` and `run-qemu.sh` expect neighboring directories `../capstone-qemu` and `../capstone-c`.

```bash
mkdir -p "$HOME/work/project-starch"
cd "$HOME/work/project-starch"

git clone https://github.com/project-starch/llvm-capstone
cd llvm-capstone
git checkout c0feaee7f78081ea58205b33b69540992e3823e2

git clone https://github.com/project-starch/caplifive-buildroot.git capstone/caplifive-buildroot
git -C capstone/caplifive-buildroot checkout 3e6dbf74f1aa5ed03e4b867fca31f27605b14afc
git -C capstone/caplifive-buildroot submodule update --init --recursive

git clone https://github.com/project-starch/capstone-qemu.git capstone/capstone-qemu
git -C capstone/capstone-qemu checkout 676f43c368297d6b6846d212f9a28a9860f44ede
git -C capstone/capstone-qemu submodule update --init --recursive

git clone https://github.com/jasonyu1996/capstone-c capstone/capstone-c
git -C capstone/capstone-c checkout 4899cf90dbd3a0d6b1b730a64dafca949e51a113
git -C capstone/capstone-c submodule update --init --recursive
```

---

## 2. Build QEMU, Capstone-C, and the Buildroot image

### 2.1 Build Capstone QEMU

```bash
cd "$HOME/work/project-starch/llvm-capstone/capstone/capstone-qemu"
bash ./local_build.sh
```

### 2.2 Build the Capstone-C compiler

If `rustc`/`cargo` are missing, the script will try to install Rust itself via `rustup`.

```bash
cd "$HOME/work/project-starch/llvm-capstone/capstone/capstone-c"
bash ./local_build.sh
```

### 2.3 Build the Buildroot system image

```bash
cd "$HOME/work/project-starch/llvm-capstone/capstone/caplifive-buildroot"
bash ./local_build.sh
```

### 2.4 Explicitly rebuild the `null_blk` case study and the userspace helper

This makes the reproduction flow more literal for the `null_blk` path specifically.

```bash
cd "$HOME/work/project-starch/llvm-capstone/capstone/caplifive-buildroot"
export CAPSTONE_CC_PATH="$(realpath ../capstone-c)"
make build CAPSTONE_CC_PATH="$CAPSTONE_CC_PATH" A=capstone-null-blk-build
make build CAPSTONE_CC_PATH="$CAPSTONE_CC_PATH" A=modcapstone-rebuild
```

---

## 3. Host-side automatic reproduction via the QEMU harness

This is the most convenient non-interactive path: QEMU boots once, the log goes to a file, and the target guest command is executed automatically.

> Important: this harness already does `insmod /capstone.ko` by itself, so you should **not** repeat `insmod /capstone.ko` inside the `--guest-command` below.

```bash
cd "$HOME/work/project-starch/llvm-capstone"
mkdir -p /tmp/capstone/capstone-runtime-qemu-share

python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --buildroot-dir "$PWD/capstone/caplifive-buildroot" \
  --qemu-binary "$PWD/capstone/capstone-qemu/build/qemu-system-riscv64" \
  --share-dir /tmp/capstone/capstone-runtime-qemu-share \
  --log-file /tmp/capstone/capstone-runtime-qemu-nullb.log \
  --guest-command "modprobe configfs && /null_blk.user && insmod /nullb/capstone_split/null_blk.ko && test -b /dev/nullb0 && echo hello-world | dd of=/dev/nullb0 bs=1024 count=10 && dd if=/dev/nullb0 bs=1024 count=1 | hexdump -C" \
  > /tmp/capstone/capstone-runtime-qemu-nullb-wrapper.txt 2>&1 || true
```

### View the wrapper log

```bash
sed -n '1,220p' /tmp/capstone/capstone-runtime-qemu-nullb-wrapper.txt
```

### View the full QEMU serial log

```bash
sed -n '1,260p' /tmp/capstone/capstone-runtime-qemu-nullb.log
```

### Quickly extract the crash signatures

```bash
grep -nE 'SBI domain created|Oops|Segmentation fault|null_add_dev|Modules linked in|exit code 139' \
  /tmp/capstone/capstone-runtime-qemu-nullb-wrapper.txt \
  /tmp/capstone/capstone-runtime-qemu-nullb.log | cat
```

---

## 4. Manual reproduction inside the guest

If you want a fully manual path without the harness:

### 4.1 Start QEMU

```bash
cd "$HOME/work/project-starch/llvm-capstone/capstone/caplifive-buildroot"
bash ./run-qemu.sh
```

### 4.2 Run the commands manually inside the guest

Login:

```text
root
```

Then in the guest shell:

```bash
modprobe configfs
insmod /capstone.ko
/null_blk.user
insmod /nullb/capstone_split/null_blk.ko
```

If the crash does not happen earlier, the README would normally expect the following commands next:

```bash
echo "hello world" | dd of=/dev/nullb0 bs=1024 count=10
dd if=/dev/nullb0 bs=1024 count=10 | hexdump -C
```

But in the currently observed state, execution usually never gets that far because the failure already happens at `insmod /nullb/capstone_split/null_blk.ko`.

---

## 5. What counts as a successful reproduction

If the reproduction worked, the logs should contain markers like these:

- `SBI domain created with ID 0`
- `Oops [#1]`
- `Modules linked in: null_blk(O+) capstone(O)`
- `epc : null_add_dev+0x38/0x740 [null_blk]`
- `Segmentation fault`
- in the wrapper path, an unsuccessful guest command exit corresponding to `exit code 139`

---

## 6. Short version of the sequence

If you strip away the explanations, the minimal command sequence is:

```bash
sudo apt-get update
sudo apt-get install -y git cmake ninja-build build-essential wget rsync curl bc libglib2.0-dev libfdt-dev libpixman-1-dev libslirp-dev libncurses5-dev libssl-dev zlib1g-dev file cpio unzip sed make binutils diffutils patch perl tar findutils bzip2 expect python3 python3-pip python3-venv python3-pexpect

mkdir -p "$HOME/work/project-starch"
cd "$HOME/work/project-starch"
git clone https://github.com/project-starch/llvm-capstone
cd llvm-capstone
git checkout c0feaee7f78081ea58205b33b69540992e3823e2
git clone https://github.com/project-starch/caplifive-buildroot.git capstone/caplifive-buildroot
git -C capstone/caplifive-buildroot checkout 3e6dbf74f1aa5ed03e4b867fca31f27605b14afc
git -C capstone/caplifive-buildroot submodule update --init --recursive
git clone https://github.com/project-starch/capstone-qemu.git capstone/capstone-qemu
git -C capstone/capstone-qemu checkout 676f43c368297d6b6846d212f9a28a9860f44ede
git -C capstone/capstone-qemu submodule update --init --recursive
git clone https://github.com/jasonyu1996/capstone-c capstone/capstone-c
git -C capstone/capstone-c checkout 4899cf90dbd3a0d6b1b730a64dafca949e51a113
git -C capstone/capstone-c submodule update --init --recursive

cd capstone/capstone-qemu
bash ./local_build.sh

cd ../capstone-c
bash ./local_build.sh

cd ../caplifive-buildroot
bash ./local_build.sh
export CAPSTONE_CC_PATH="$(realpath ../capstone-c)"
make build CAPSTONE_CC_PATH="$CAPSTONE_CC_PATH" A=capstone-null-blk-build
make build CAPSTONE_CC_PATH="$CAPSTONE_CC_PATH" A=modcapstone-rebuild

cd "$HOME/work/project-starch/llvm-capstone"
mkdir -p /tmp/capstone/capstone-runtime-qemu-share
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --buildroot-dir "$PWD/capstone/caplifive-buildroot" \
  --qemu-binary "$PWD/capstone/capstone-qemu/build/qemu-system-riscv64" \
  --share-dir /tmp/capstone/capstone-runtime-qemu-share \
  --log-file /tmp/capstone/capstone-runtime-qemu-nullb.log \
  --guest-command "modprobe configfs && /null_blk.user && insmod /nullb/capstone_split/null_blk.ko && test -b /dev/nullb0 && echo hello-world | dd of=/dev/nullb0 bs=1024 count=10 && dd if=/dev/nullb0 bs=1024 count=1 | hexdump -C" \
  > /tmp/capstone/capstone-runtime-qemu-nullb-wrapper.txt 2>&1 || true

sed -n '1,220p' /tmp/capstone/capstone-runtime-qemu-nullb-wrapper.txt
sed -n '1,260p' /tmp/capstone/capstone-runtime-qemu-nullb.log
grep -nE 'SBI domain created|Oops|Segmentation fault|null_add_dev|Modules linked in|exit code 139' /tmp/capstone/capstone-runtime-qemu-nullb-wrapper.txt /tmp/capstone/capstone-runtime-qemu-nullb.log | cat
```

---

## 7. Where this comes from

This sequence was assembled from existing source files and handoff notes already present in this workspace:

- `capstone/caplifive-buildroot/README.md`
- `capstone/caplifive-buildroot/local_build.sh`
- `capstone/capstone-qemu/local_build.sh`
- `capstone/capstone-c/local_build.sh`
- `capstone/tests/runtime-qemu/run-domain-smoke.py`
- `capstone/agent-handoff/history/2026-05-08T12-19-23Z-null-block-reference-test.md`

That is why this document includes all of the following in one place:
- the manual guest-side path,
- the automated QEMU-harness path,
- exact repository URLs,
- pinned commits,
- and the expected crash signatures.

