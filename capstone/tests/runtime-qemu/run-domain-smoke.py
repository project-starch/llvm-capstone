#!/usr/bin/env python3

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Iterable, Optional

import pexpect

INFRA_FLAKE_MARKER = "__CAPSTONE_INFRA_FLAKE__"

# THE PROMPT. Busybox's root prompt is "# ", and "# " also turns up in what a
# guest command prints and in the echo of what was typed: a comment in an import
# trace, a sed expression, a shell comment. Each such line used to be taken for
# the prompt, and the runner then typed its exit-code probe into the running
# program and failed the run ("could not parse exit code", or a 40 s timeout that
# read like a guest stall). Right after login the runner sets PS1 to this marker
# and from then on waits for it and nothing else. It types the assignment with
# the marker split across two quoted halves, so the echo of that line does not
# contain the marker either.
PROMPT_MARKER = "__CAPSTONE_PROMPT__: "
_prompt_pattern = re.escape("# ")

# LONG COMMANDS. The guest shell's line editor keeps about 1 KiB of a typed line
# (1022 characters of a longer one arrived on 2026-09-23, the rest was dropped and
# the shell sat at its continuation prompt). A guest command longer than this is
# written to a file on the share and SOURCED there (".", not "sh"), so it runs in
# the same shell with the same state, and its status is the status it would have
# had typed.
LONG_COMMAND = 800
_share_dir: Optional[pathlib.Path] = None
_share_mounted = False
_command_files = 0
INFRA_FLAKE_EXIT_CODE = 75

DEFAULT_DOMAIN_SUCCESS_MARKERS = (
    "Ok, good file.",
    "Loadable executable segment found.",
    "Created domain ID = 0",
    "Called dom (1-th time) retval =",
)


def env_or_default(name: str, default: pathlib.Path | str) -> str:
    return os.environ.get(name, str(default))


def env_float_or_default(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number, got {value!r}") from exc
    if parsed <= 0:
        raise SystemExit(f"{name} must be positive, got {value!r}")
    return parsed


class NormalizedLogWriter:
    """Turns the console's line endings into "\n". pexpect hands over whatever one
    read returned, so a "\r\n" can arrive split: the "\r" at the end of one chunk,
    the "\n" at the start of the next. Normalising each chunk on its own then
    writes two newlines for one line end, a blank line at a random place in the
    log. The trailing "\r"s are held back until the next chunk shows what follows
    them. flush() does not emit them: pexpect flushes after every write."""

    def __init__(self, sink):
        self.sink = sink
        self.held = ""

    def write(self, data: str) -> None:
        data = self.held + data
        kept = data.rstrip("\r")
        self.held = data[len(kept):]
        normalized = kept.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
        self.sink.write(normalized)

    def flush(self) -> None:
        self.sink.flush()


class InfraFlakeError(RuntimeError):
    def __init__(self, phase: str, message: str):
        self.phase = phase
        super().__init__(message)


def parse_args() -> argparse.Namespace:
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    default_tmp_root = pathlib.Path(env_or_default("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
    buildroot_dir = pathlib.Path(
        env_or_default("CAPSTONE_BUILDROOT_DIR", repo_root / "capstone" / "caplifive-buildroot")
    )
    qemu_bin = pathlib.Path(
        env_or_default(
            "CAPSTONE_QEMU_BINARY",
            repo_root / "capstone" / "capstone-qemu" / "build" / "qemu-system-riscv64",
        )
    )

    parser = argparse.ArgumentParser(
        description="Boot Capstone QEMU once, mount a host-shared 9p directory, and run domain smoke tests."
    )
    parser.add_argument("domains", nargs="*", help="Domain ELF files to run inside the guest.")
    parser.add_argument(
        "--guest-command",
        action="append",
        default=[],
        help="Additional shell command to run inside the guest after the standard setup.",
    )
    parser.add_argument(
        "--success-marker",
        action="append",
        default=[],
        help="Success marker required in each --guest-command output.",
    )
    parser.add_argument(
        "--share-dir",
        default=str(default_tmp_root / "capstone-runtime-qemu-share"),
        help="Host directory exported into the guest over 9p.",
    )
    parser.add_argument(
        "--log-file",
        default=str(default_tmp_root / "capstone-runtime-qemu-smoke.log"),
        help="Full serial/QEMU log file.",
    )
    parser.add_argument(
        "--buildroot-dir",
        default=str(buildroot_dir),
        help="Path to capstone/caplifive-buildroot.",
    )
    parser.add_argument(
        "--qemu-binary",
        default=str(qemu_bin),
        help="Path to qemu-system-riscv64 built from capstone-qemu.",
    )
    parser.add_argument(
        "--domain-loader",
        default="/capstone-test.user",
        help="Guest-visible path to the domain loader binary used for positional domain arguments.",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Multiply the default expect timeouts by this factor.",
    )
    parser.add_argument(
        "--qemu-extra-arg",
        action="append",
        default=[],
        help="Extra argument appended verbatim to the qemu command line "
        "(repeatable). Additive; used e.g. by the borrow-cost probe to pass "
        "-icount for a deterministic instruction count.",
    )
    parser.add_argument(
        "--kernel-arg",
        action="append",
        default=[],
        help="Extra word for the guest kernel's command line (repeatable), "
        "e.g. cma=1G: the CMA area a region above the buddy allocator's 4 MiB "
        "block comes from (modcapstone create_region).",
    )
    args = parser.parse_args()
    if not args.domains and not args.guest_command:
        parser.error("provide at least one domain path or one --guest-command")
    return args


def last_exit_code(text: str) -> int:
    for line in reversed(text.replace("\r\r", "\r").splitlines()):
        stripped = line.strip()
        if stripped.startswith("__EXIT_CODE__"):
            return int(stripped.removeprefix("__EXIT_CODE__"))
    raise RuntimeError(f"could not parse exit code from: {text!r}")


def serial_tail(qemu: pexpect.spawn, limit: int = 60) -> str:
    text = qemu.before.replace("\r\r", "\r").strip()
    if not text:
        return "<no serial output captured>"
    lines = text.splitlines()
    return "\n".join(lines[-limit:])


def expect_prompt(
    qemu: pexpect.spawn, *, timeout: float, action: str, infra_phase: Optional[str] = None
) -> None:
    try:
        qemu.expect(_prompt_pattern, timeout=timeout)
    except (pexpect.EOF, pexpect.TIMEOUT) as exc:
        message = (
            f"QEMU stopped before the shell prompt while {action}.\n"
            f"Recent serial output:\n{serial_tail(qemu)}"
        )
        if infra_phase is not None:
            raise InfraFlakeError(infra_phase, message) from exc
        raise RuntimeError(message) from exc


def undefined_weak_symbols(share_dir: pathlib.Path) -> dict[str, list[str]]:
    """Every .dom in the share that references an undefined weak symbol.

    In a domain the address of an undefined weak symbol is NOT NULL (C-56): it is
    formed pc-relative against gp, the linker resolves the symbol to 0 at the link
    address, and the image runs at another base without relocation. So
    `if (&sym)` is true and a call through it jumps into the image. An image
    carrying one is refused before it costs a boot; CAPSTONE_ALLOW_UNDEFINED_WEAK=1
    lets a test that wants one through (a C-56 reproducer, say).
    """
    nm = None
    for cand in (os.environ.get("CAPSTONE_LLVM_BIN"),
                 os.environ.get("CAPSTONE_LLVM_BUILD_DIR") and
                 os.path.join(os.environ["CAPSTONE_LLVM_BUILD_DIR"], "bin")):
        if cand and os.path.exists(os.path.join(cand, "llvm-nm")):
            nm = os.path.join(cand, "llvm-nm")
            break
    nm = nm or shutil.which("llvm-nm")
    if nm is None:
        raise RuntimeError(
            "undefined-weak preflight: no llvm-nm (set CAPSTONE_LLVM_BIN or "
            "CAPSTONE_LLVM_BUILD_DIR); refusing to report a clean share it did not check"
        )
    found: dict[str, list[str]] = {}
    for image in sorted(share_dir.glob("*.dom")):
        out = subprocess.run([nm, "--undefined-only", str(image)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"undefined-weak preflight: {nm} failed on {image}: {out.stderr.strip()}")
        weak = [line.split()[-1] for line in out.stdout.splitlines()
                if line.split() and line.split()[0] in ("w", "v")]
        if weak:
            found[image.name] = weak
    return found


def copy_domains_into_share(domains: Iterable[str], share_dir: pathlib.Path) -> list[str]:
    share_dir.mkdir(parents=True, exist_ok=True)
    guest_names = []
    for domain in domains:
        src = pathlib.Path(domain).resolve()
        dst = share_dir / src.name
        if src != dst:
            shutil.copy2(src, dst)
        guest_names.append(src.name)
    return guest_names


def run_guest_command(
    qemu: pexpect.spawn,
    command: str,
    timeout: float = 20.0,
    *,
    infra_phase: Optional[str] = None,
) -> str:
    global _command_files
    typed = command
    if len(command) > LONG_COMMAND and _share_mounted and _share_dir is not None:
        _command_files += 1
        name = f".runner-command-{_command_files}.sh"
        (_share_dir / name).write_text(command + "\n")
        typed = f". /mnt/host/{name}"
    qemu.sendline(typed)
    expect_prompt(
        qemu,
        timeout=timeout,
        action=f"running guest command: {command}",
        infra_phase=infra_phase,
    )
    output = qemu.before.replace("\r\r", "\r")

    qemu.sendline("printf '__EXIT_CODE__%s\\n' $?")
    expect_prompt(
        qemu,
        timeout=5,
        action="collecting the previous guest exit code",
        infra_phase=infra_phase,
    )
    exit_capture = qemu.before.replace("\r\r", "\r")
    exit_code = last_exit_code(exit_capture)
    if exit_code != 0:
        message = (
            f"guest command failed with exit code {exit_code}: {command}\n{output}\n{exit_capture}"
        )
        if infra_phase is not None:
            raise InfraFlakeError(infra_phase, message)
        raise RuntimeError(message)
    return output


def power_off(qemu: pexpect.spawn) -> None:
    try:
        qemu.sendline("poweroff -f")
        qemu.expect(pexpect.EOF, timeout=20)
    except (pexpect.EOF, pexpect.TIMEOUT):
        pass
    finally:
        qemu.terminate(force=True)


def main() -> int:
    args = parse_args()
    share_dir = pathlib.Path(args.share_dir).resolve()
    log_file = pathlib.Path(args.log_file).resolve()
    buildroot_dir = pathlib.Path(args.buildroot_dir).resolve()
    qemu_binary = pathlib.Path(args.qemu_binary).resolve()
    domain_loader = args.domain_loader

    if not qemu_binary.exists():
        raise SystemExit(f"missing qemu binary: {qemu_binary}")

    image_dir = buildroot_dir / "build" / "images"
    fw_jump = image_dir / "fw_jump.elf"
    kernel = image_dir / "Image"
    rootfs = image_dir / "rootfs.ext2"
    for required in (fw_jump, kernel, rootfs):
        if not required.exists():
            raise SystemExit(f"missing buildroot image: {required}")

    guest_domains = copy_domains_into_share(args.domains, share_dir)
    global _share_dir, _share_mounted, _prompt_pattern
    _share_dir = share_dir
    if os.environ.get("CAPSTONE_ALLOW_UNDEFINED_WEAK") != "1":
        weak = undefined_weak_symbols(share_dir)
        if weak:
            detail = "\n".join(f"  {name}: {', '.join(syms)}" for name, syms in weak.items())
            raise RuntimeError(
                "undefined weak symbols in domain images (their address is not NULL in a "
                "domain, C-56; define them, or set CAPSTONE_ALLOW_UNDEFINED_WEAK=1):\n" + detail
            )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    timeout_multiplier = args.timeout_multiplier
    qemu_cmd = [
        str(qemu_binary),
        "-M",
        "virt-capstone",
        "-m",
        "8G",
        "-smp",
        "1",
        "-nographic",
        "-bios",
        str(fw_jump),
        "-kernel",
        str(kernel),
        "-append",
        " ".join(["root=/dev/vda ro"] + args.kernel_arg),
        "-snapshot",
        "-drive",
        f"file={rootfs},format=raw,id=hd0",
        "-device",
        "virtio-blk-device,drive=hd0",
        "-virtfs",
        f"local,path={share_dir},mount_tag=hostshare,security_model=none,id=hostshare",
        "-chardev",
        "stdio,mux=on,id=ch0,signal=on",
        "-mon",
        "chardev=ch0,mode=readline",
        "-serial",
        "chardev:ch0",
        "-cpu",
        "rv64,sstc=false,h=false",
    ]
    qemu_cmd.extend(args.qemu_extra_arg)

    with log_file.open("w", encoding="utf-8") as log:
        normalized_log = NormalizedLogWriter(log)
        log.write("> starting qemu with:\n")
        log.write(" ".join(qemu_cmd) + "\n\n")
        qemu = pexpect.spawn(
            qemu_cmd[0],
            qemu_cmd[1:],
            timeout=max(5.0, 5.0 * timeout_multiplier),
            encoding="utf-8",
            codec_errors="replace",
            cwd=str(buildroot_dir),
            env={"QEMU_AUDIO_DRV": "none", **os.environ},
        )
        qemu.logfile_read = normalized_log

        try:
            try:
                login_timeout = env_float_or_default(
                    "CAPSTONE_QEMU_LOGIN_TIMEOUT", 120 * timeout_multiplier
                )
                qemu.expect("buildroot login:", timeout=login_timeout)
            except (pexpect.EOF, pexpect.TIMEOUT) as exc:
                raise InfraFlakeError(
                    "boot-login",
                    "QEMU stopped before the guest login prompt appeared.\n"
                    f"Recent serial output:\n{serial_tail(qemu)}",
                ) from exc
            qemu.sendline("root")
            expect_prompt(
                qemu,
                timeout=30 * timeout_multiplier,
                action="logging into the guest",
                infra_phase="guest-login",
            )
            # From here on the prompt is the marker (see PROMPT_MARKER).
            half = len(PROMPT_MARKER) // 2
            qemu.sendline(f"PS1='{PROMPT_MARKER[:half]}''{PROMPT_MARKER[half:]}'")
            _prompt_pattern = re.escape(PROMPT_MARKER)
            expect_prompt(
                qemu,
                timeout=30 * timeout_multiplier,
                action="setting the prompt marker",
                infra_phase="guest-login",
            )

            run_guest_command(
                qemu, "dmesg -n 1", timeout=10 * timeout_multiplier, infra_phase="guest-setup"
            )
            run_guest_command(
                qemu,
                "stty columns 29999",
                timeout=10 * timeout_multiplier,
                infra_phase="guest-setup",
            )
            run_guest_command(
                qemu,
                "mkdir -p /mnt/host",
                timeout=10 * timeout_multiplier,
                infra_phase="guest-setup",
            )
            run_guest_command(
                qemu,
                "mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/host",
                timeout=20 * timeout_multiplier,
                infra_phase="guest-setup",
            )
            _share_mounted = True
            run_guest_command(
                qemu,
                "insmod /capstone.ko",
                timeout=20 * timeout_multiplier,
                infra_phase="guest-setup",
            )

            # THE WORKLOAD GETS ITS OWN BUDGET, separate from the multiplier.
            #
            # --timeout-multiplier scales EVERY timeout, including the ones that
            # should stay short. A caller whose workload needs 45 minutes used to
            # have to raise the multiplier, which also gave a hung BOOT 45
            # minutes before it was called a flake -- so one bad boot cost the
            # session most of an hour and produced nothing. Setup should fail
            # fast; only the workload should be allowed to be slow.
            command_timeout = env_float_or_default(
                "CAPSTONE_GUEST_COMMAND_TIMEOUT", 30 * timeout_multiplier
            )
            for domain_name in guest_domains:
                output = run_guest_command(
                    qemu,
                    f"{domain_loader} /mnt/host/{domain_name}",
                    timeout=command_timeout,
                )
                missing = [marker for marker in DEFAULT_DOMAIN_SUCCESS_MARKERS if marker not in output]
                if missing:
                    raise RuntimeError(
                        f"domain smoke missing success markers for {domain_name}: {missing}\n{output}"
                    )

            for guest_command in args.guest_command:
                output = run_guest_command(
                    qemu,
                    guest_command,
                    timeout=command_timeout,
                )
                if args.success_marker:
                    missing = [marker for marker in args.success_marker if marker not in output]
                    if missing:
                        raise RuntimeError(
                            f"guest command missing success markers: {missing}\n"
                            f"command: {guest_command}\n{output}"
                        )

            power_off(qemu)
        except Exception:
            qemu.terminate(force=True)
            raise

    print("QEMU smoke passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InfraFlakeError as exc:
        print(f"{INFRA_FLAKE_MARKER} phase={exc.phase}", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(INFRA_FLAKE_EXIT_CODE)
