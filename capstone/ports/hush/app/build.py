#!/usr/bin/env python3
"""Build pinned hush using either a native CC or the common capability SDK."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ENABLED = """
HUSH SHELL_HUSH SH_IS_HUSH HUSH_LINENO_VAR HUSH_TICK HUSH_IF HUSH_LOOPS
HUSH_CASE HUSH_FUNCTIONS HUSH_LOCAL HUSH_MODE_X HUSH_ECHO HUSH_PRINTF HUSH_TEST
HUSH_EXPORT HUSH_READONLY HUSH_WAIT HUSH_TYPE HUSH_READ HUSH_SET HUSH_UNSET
HUSH_GETOPTS FEATURE_SH_STANDALONE FEATURE_SH_MATH FEATURE_SH_MATH_64
ECHO PRINTF TEST TEST1 TEST2 CAT TRUE FALSE FEATURE_FANCY_ECHO
""".split()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--sdk", type=Path, help="common SDK containing capstone-cc and sdk.json")
    p.add_argument("--cc", default="cc", help="host compiler (without --sdk)")
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    out = args.output.resolve()
    tmp = Path(os.environ["CAPSTONE_TMP_ROOT"]).resolve()
    if tmp not in out.parents or out.exists():
        p.error("output must be a fresh directory below CAPSTONE_TMP_ROOT")
    out.mkdir(parents=True)
    (out / "tmp").mkdir()
    os.environ["TMPDIR"] = str(out / "tmp")
    src = Path(subprocess.check_output(["bash", str(HERE / "prepare-sources.sh"),
                                      str(out / "source")], text=True).strip())
    build = out / "build"
    build.mkdir()
    target = out / ("hush.dom" if args.sdk else "busybox")
    cc = str(args.sdk.resolve() / "capstone-cc") if args.sdk else args.cc
    sdk_cc = cc
    if args.sdk:
        wrapper = out / "cc"
        shutil.copy2(HERE / "sdk-cc.py", wrapper)
        wrapper.chmod(0o755)
        os.environ["CAPSTONE_HUSH_SDK_CC"] = cc
        cc = str(wrapper)
    command = ["make", "-C", str(src), "O=" + str(build), "CC=" + cc,
               "HOSTCC=cc", "AR=ar", "STRIP=true"]
    if args.sdk:
        metadata = json.loads((args.sdk / "sdk.json").read_text())
        # BusyBox combines applet objects with a relocatable link. The common
        # SDK driver owns final image links; invoke its recorded linker for -r.
        command.append("LD=" + metadata["ld"])
    with (out / "build.log").open("w") as log:
        def make(*goals):
            subprocess.run(command + list(goals), check=True, stdout=log,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        make("allnoconfig")
        config_path = build / ".config"
        config = config_path.read_text()
        for name in ENABLED:
            pattern = r"^(?:# )?CONFIG_" + name + r"(?:=.*| is not set)$"
            config, count = re.subn(pattern, "CONFIG_" + name + "=y", config, flags=re.M)
            if count != 1:
                raise RuntimeError("missing config option " + name)
        config = config.replace("CONFIG_SH_IS_NONE=y", "# CONFIG_SH_IS_NONE is not set")
        config = config.replace("CONFIG_SH_IS_ASH=y", "# CONFIG_SH_IS_ASH is not set")
        exec_path = "/bin/sh" if args.sdk else str(target)
        config = re.sub(r"^CONFIG_BUSYBOX_EXEC_PATH=.*$",
                        "CONFIG_BUSYBOX_EXEC_PATH=" + json.dumps(exec_path), config, flags=re.M)
        # Upstream documents this escape hatch for toolchains that hoist reads
        # of its intentionally mutable 'const' global pointers. Such a hoist
        # loads an untagged null capability before INIT_G stores the allocation.
        config = re.sub(r"^CONFIG_EXTRA_CFLAGS=.*$",
                        'CONFIG_EXTRA_CFLAGS="-DBB_GLOBAL_CONST="', config, flags=re.M)
        config_path.write_text(config)
        make("oldconfig")
        make("-j" + str(args.jobs), "busybox_unstripped")
    shutil.copy2(build / "busybox_unstripped", target)
    manifest = {
        "upstream": "BusyBox 1.37.0 hush",
        "archiveSha256": digest(out / "source/busybox-1.37.0.tar.bz2"),
        "patches": {f.name: digest(f) for f in sorted((HERE / "patches").glob("*.patch"))},
        "configSha256": digest(config_path), "image": str(target),
        "imageSha256": digest(target), "cc": sdk_cc,
        "sdk": json.loads((args.sdk / "sdk.json").read_text()) if args.sdk else None,
        "shellExecPath": exec_path,
        "scope": "noninteractive spawn shell; no caught traps, job control or in-place exec",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
