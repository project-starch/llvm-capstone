#!/usr/bin/env python3
"""Build mechanism-probe.c with the virtual SDK and run it in one boot of the kit: the type sizes that decide the
CWE-122 sizeof family, read in the same guest and ABI as the cases.

    probe.py --sdk <virtual SDK dir> --kit <kit> --helper <juliet-timeout> --out <fresh dir>   -> <out>/probe.json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import common as C
import virtualvm  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--helper", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; a probe goes into a fresh directory")
    img = a.out / "img"; img.mkdir(parents=True)
    dom = img / "mechanism-probe.dom"
    subprocess.run([str(a.sdk / "capstone-cc"), "-O0", str(C.HERE / "mechanism-probe.c"), "-o", str(dom)], check=True,
                   env=C.clean_env())
    b = virtualvm.Batch(a.out / "stage")
    b.put("images", img)
    b.put("juliet-timeout", a.helper)
    b.step("probe", "./juliet-timeout probe 10000 -- ./capstone-vexec ./images/mechanism-probe.dom")
    steps, serial, completed = b.execute(a.kit, a.out / "work", 300)
    text, result = steps.get("probe", ("", {"kind": "none"}))
    lines = re.findall(r"^PROBE (\S+) (.*)$", text, re.M)
    record = {"completed": completed, "exit": [result.get("kind"), result.get("value")],
              "platform": virtualvm.platform(a.kit, None, C.HERE / "run.py"), "sdk_fingerprint": C.sdk_fingerprint(a.sdk),
              "image_sha256": C.sha256(dom), "source_sha256": C.sha256(C.HERE / "mechanism-probe.c"),
              "readings": {k: dict(re.findall(r"(\S+)=(\S+)", v)) for k, v in lines}}
    (a.out / "probe.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record["readings"]))
    sys.exit(0 if completed and record["exit"] == ["exit", 0] and "sizeof-ptr" in record["readings"] else 75)


if __name__ == "__main__":
    main()
