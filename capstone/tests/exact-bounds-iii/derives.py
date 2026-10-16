#!/usr/bin/env python3
"""Pull each capstone_cap_derive(...) call's OFFSET and LENGTH arguments out of the ports' patches,
so each can be bounded from code. Calls span lines, so the text is joined and the parentheses
balanced rather than grepped line by line."""
import re, subprocess, sys

FILES = [
 "capstone/ports/ffmpeg/buffer-pool/patches/ffmpeg-9.0.1-0003-libavutil-pools-as-sublet-lifetimes.patch",
 "capstone/ports/memcached/app/patches/memcached-1.6.45-0006-slabs-and-cache-sublet-lifetimes.patch",
 "capstone/ports/wireshark/wmem/patches/wireshark-4.6.8-0001-wmem-sublet-lifetimes.patch",
 "capstone/ports/postgres/memory-contexts/patches/postgresql-17.0-0003-memory-contexts-sublet-lifetimes.patch",
 "capstone/ports/sqlite/sublet/sublet-3220000-memsys5.patch",
]

def args_of(text, start):
    """The argument list of the call whose '(' follows `start`, with parentheses balanced."""
    i = text.index("(", start); depth = 0; out = []
    for j in range(i, len(text)):
        c = text[j]
        if c == "(": depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                inner = text[i+1:j]
                break
    else:
        return None
    # split on top-level commas
    depth = 0; cur = ""; parts = []
    for c in inner:
        if c == "(": depth += 1
        if c == ")": depth -= 1
        if c == "," and depth == 0:
            parts.append(cur.strip()); cur = ""
        else:
            cur += c
    parts.append(cur.strip())
    return parts

for f in FILES:
    blob = subprocess.run(["git", "show", "origin/dev:" + f], capture_output=True, text=True).stdout
    added = "\n".join(l[1:] for l in blob.split("\n") if l.startswith("+") and not l.startswith("+++"))
    flat = re.sub(r"\s+", " ", added)
    print("=" * 100)
    print(f.split("/")[-1])
    for m in re.finditer(r"capstone_cap_derive", flat):
        a = args_of(flat, m.start())
        if not a or len(a) < 3:
            print("   (unparsed)"); continue
        print("   parent=%-34s offset=%-44s LEN=%s" % (a[0][:34], a[1][:44], a[2][:60]))
