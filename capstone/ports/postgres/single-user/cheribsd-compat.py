#!/usr/bin/env python3
"""Adapt the shared PostgreSQL capability-alignment patch to CHERI intcap."""

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {old!r}")
    return text.replace(old, new)


def main() -> None:
    header = Path(sys.argv[1])
    text = header.read_text()
    text = replace_once(
        text,
        "\tlong long: 0, unsigned long long: 0, \\\n\tdefault: pg_typealign_bad_type())",
        "\tlong long: 0, unsigned long long: 0, \\\n\t__intcap_t: 0, __uintcap_t: 0, \\\n\tdefault: pg_typealign_bad_type())",
    )
    text = replace_once(
        text,
        "#define PG_TYPEALIGN_AS_INT(X) _Generic((X), \\\n"
        "\tchar *: (uintptr_t) 0, \\\n"
        "\tconst char *: (uintptr_t) 0, \\\n"
        "\tunsigned char *: (uintptr_t) 0, \\\n"
        "\tconst unsigned char *: (uintptr_t) 0, \\\n"
        "\tsigned char *: (uintptr_t) 0, \\\n"
        "\tvoid *: (uintptr_t) 0, \\\n"
        "\tconst void *: (uintptr_t) 0, \\\n"
        "\tdefault: (X))",
        "#define PG_TYPEALIGN_AS_INT(X) _Generic((X), \\\n"
        "\tchar *: (unsigned long) 0, \\\n"
        "\tconst char *: (unsigned long) 0, \\\n"
        "\tunsigned char *: (unsigned long) 0, \\\n"
        "\tconst unsigned char *: (unsigned long) 0, \\\n"
        "\tsigned char *: (unsigned long) 0, \\\n"
        "\tvoid *: (unsigned long) 0, \\\n"
        "\tconst void *: (unsigned long) 0, \\\n"
        "\tdefault: (X))",
    )
    text = replace_once(
        text,
        "\t\tdefault: (((uintptr_t) PG_TYPEALIGN_AS_INT(LEN) + ((ALIGNVAL) - 1)) & ~((uintptr_t) ((ALIGNVAL) - 1))) \\\n\t\t\t+ PG_TYPEALIGN_INT(LEN))",
        "\t\tdefault: (PG_TYPEALIGN_AS_INT(LEN) + \\\n\t\t\t(((ALIGNVAL) - ((unsigned long) PG_TYPEALIGN_AS_INT(LEN) & ((ALIGNVAL) - 1))) & ((ALIGNVAL) - 1)) \\\n\t\t\t+ PG_TYPEALIGN_INT(LEN)))",
    )
    text = replace_once(
        text,
        "\t\tdefault: (((uintptr_t) PG_TYPEALIGN_AS_INT(LEN)) & ~((uintptr_t) ((ALIGNVAL) - 1))) \\\n\t\t\t+ PG_TYPEALIGN_INT(LEN))",
        "\t\tdefault: (PG_TYPEALIGN_AS_INT(LEN) - \\\n\t\t\t((unsigned long) PG_TYPEALIGN_AS_INT(LEN) & ((ALIGNVAL) - 1)) \\\n\t\t\t+ PG_TYPEALIGN_INT(LEN)))",
    )
    header.write_text(text)


if __name__ == "__main__":
    main()
