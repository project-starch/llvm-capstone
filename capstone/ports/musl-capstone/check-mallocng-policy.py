#!/usr/bin/env python3
"""Check allocator policy sources against a verified upstream release.

Platform glue is deliberately separate. This gate does not establish capability
safety or qualify an allocator that has not yet been integrated.
"""
import argparse
import hashlib
from pathlib import Path
import tarfile


RELEASE_SHA256 = "a9a118bbe84d8764da0ea0d28b3ab3fae8477fc7e4085d90102b8596fc7c75e4"
PREFIX = "src/malloc/mallocng/"


def check(source, archive, expected, include_glue=False):
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("musl archive SHA-256 mismatch")
    checked = []
    errors = []
    with tarfile.open(archive) as release:
        for member in release.getmembers():
            relative = member.name.partition("/")[2]
            if not member.isfile() or not relative.startswith(PREFIX):
                continue
            name = Path(relative).name
            if (name == "glue.h" and not include_glue) or Path(name).suffix not in (".c", ".h"):
                continue
            original = release.extractfile(member).read()
            path = source / relative
            if not path.is_file() or path.read_bytes() != original:
                errors.append(relative)
            checked.append(relative)
    if PREFIX + "meta.h" not in checked or PREFIX + "realloc.c" not in checked:
        raise ValueError("release does not contain the expected mallocng policy sources")
    # An added policy source is also a change; it must not escape the comparison.
    for path in (source / PREFIX).iterdir():
        relative = path.relative_to(source).as_posix()
        if path.suffix in (".c", ".h") and (include_glue or path.name != "glue.h") and relative not in checked:
            errors.append(relative)
    if errors:
        raise ValueError("allocator policy differs from upstream:\n" + "\n".join(sorted(errors)))
    return len(checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", default=RELEASE_SHA256)
    parser.add_argument("--include-glue", action="store_true", help="Require the complete native allocator to match")
    args = parser.parse_args()
    try:
        count = check(args.source, args.archive, args.sha256, args.include_glue)
    except (ValueError, OSError, tarfile.TarError) as error:
        parser.exit(1, f"{error}\n")
    boundary = "including glue.h" if args.include_glue else "glue.h excluded"
    print(f"PASS: {count} mallocng policy sources match the verified release; {boundary}")


if __name__ == "__main__":
    main()
