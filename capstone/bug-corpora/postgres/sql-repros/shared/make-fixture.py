#!/usr/bin/env python3
"""Create the extensions a case needs INTO the fixture cluster, on an arm that can.

    make-fixture.py --state <vm state> --image <postgres.dom> --fixture <in>
                    --out <new fixture> --extensions ltree[,pg_trgm...]

WHY THIS EXISTS, and what it does not claim.

Case 03 reaches its defect through ltree's lquery parser. Its trigger therefore
begins `CREATE EXTENSION ltree`, and on the sublet arm that statement takes a
capability fault before any of the case's own SQL runs -- so the arm scored
`not-applicable` and the defect was never shown to the mechanism.

Creating the extension is setup, not the defect. Doing it once on an arm that
can, and handing the resulting cluster to the arm that cannot, lets the second
arm run the statement the case is actually about. That is the whole purpose.

IT DOES NOT FIX THE FAULT. The sublet arm still cannot create ltree, and that
is worth reporting on its own: a capability fault while loading an extension
is a finding about the port, not about any defect in this corpus. A result
obtained through this script must say that the extension was pre-created, so a
reader does not take it as evidence that the arm handles extension creation.

The two arms are the same backend source and the same catalog version,
differing only in the Sublet patch, so a catalog written by one is readable by
the other. An ordinary host cluster is NOT interchangeable with these: the
domain builds force MAXIMUM_ALIGNOF to 16 and the on-disk layout differs.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
PYTHON = sys.executable


def vm(state, *argv, timeout=300):
    return subprocess.run(
        [PYTHON, "-m", "capstone_vm", "--state", str(state), *argv],
        cwd=str(REPO / "capstone/runtime/host"), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extensions", required=True)
    ap.add_argument("--extension-files", type=Path,
                    help="defaults to <image>/../../share/extension")
    args = ap.parse_args()
    names = [e.strip() for e in args.extensions.split(",") if e.strip()]
    args.extension_files = args.extension_files or (
        args.image.resolve().parents[1] / "share" / "extension")

    if args.out.exists():
        sys.exit(f"{args.out} exists; refusing to overwrite a fixture in place")
    config = json.loads((args.state / "config.json").read_text())
    share = Path(config["share"])
    if "running" not in vm(args.state, "status").stdout:
        sys.exit("the VM is not up")

    # The same staging the runner does, for the same reason: CREATE EXTENSION
    # reads the control file before it asks dfmgr for the library.
    target = share / "pgshare" / "extension"
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(args.extension_files.iterdir()):
        shutil.copy2(src, target / src.name)
    vm(args.state, "exec", "--", "/bin/sh", "-c",
       'grep -q "^pg:" /etc/passwd || echo "pg:x:1000:1000:pg:/tmp:/bin/sh" >> /etc/passwd; '
       'grep -q "^pg:" /etc/group  || echo "pg:x:1000:" >> /etc/group')
    staged = vm(args.state, "exec", "--", "/bin/sh", "-c",
                "mkdir -p /usr/local/pgsql && "
                "[ -d /usr/local/pgsql/share/timezonesets ] || "
                "cp -a /mnt/host/pgshare /usr/local/pgsql/share; "
                "cp -f /mnt/host/pgshare/extension/* "
                "       /usr/local/pgsql/share/extension/ 2>/dev/null; "
                "test -d /usr/local/pgsql/share/timezonesets && echo SHARE-OK")
    if "SHARE-OK" not in staged.stdout:
        sys.exit(f"guest share staging failed: {staged.stdout[-400:]}")

    work = share / "pgdata-mkfixture"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(args.fixture, work)
    subprocess.run(["chmod", "-R", "u+rwX", str(work)])
    subprocess.run(["chown", "-R", "1000:1000", str(work)], stderr=subprocess.DEVNULL)

    sql = share / "mkfixture.sql"
    # One statement per line: the stand-alone backend has no continuation. The
    # readback is what proves the extension exists, because an ERROR does not
    # end the session and a marker printed after the CREATE would print anyway.
    sql.write_text("".join(f"CREATE EXTENSION {n};\n" for n in names)
                   + "SELECT 'MADE-' || extname AS marker FROM pg_extension"
                   + " WHERE extname IN (" + ", ".join(f"'{n}'" for n in names) + ");\n")
    log = args.out.with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "run.py", "--state", str(args.state), "--cwd", "/tmp",
               "--user", "1000:1000", "--result", str(args.out.with_suffix(".json")),
               "--stdin", str(sql), str(args.image),
               "--", "--single", "-D", "/mnt/host/pgdata-mkfixture",
               "-c", "shared_buffers=4MB", "-c", "max_connections=10",
               "-c", "timezone=GMT", "-c", "log_timezone=GMT",
               "-c", "dynamic_shared_memory_type=sysv", "postgres"]
    with log.open("w") as fh:
        subprocess.run(command, cwd=str(REPO / "capstone/ports/common/application"),
                       stdout=fh, stderr=subprocess.STDOUT, timeout=1800)
    text = log.read_text(errors="replace")

    missing = [n for n in names if f"MADE-{n}" not in text]
    if missing:
        shutil.rmtree(work, ignore_errors=True)
        sys.exit(f"not created: {', '.join(missing)} -- see {log}")

    shutil.move(str(work), str(args.out))
    subprocess.run(["chmod", "-R", "u+rwX", str(args.out)])
    print(f"fixture with {', '.join(names)} written to {args.out}")
    print(f"  built by {args.image}")
    print(f"  log {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
