#!/usr/bin/env python3
"""Instrument _zval_ptr_dtor, which is where ASAN reports CRASH-047's use-after-free.

The question a completing run cannot answer: was the zval being decremented ever FREED in this
build? If it was, the port reproduces the defect and the miss is real. If it was not, the object is
simply not individually allocated here and there is nothing for revoke-on-free to key on.

  usage: patch-zend-execute-api.py <in.c> <out.c>
"""
import sys

# The bare `(*zval_ptr)->refcount--;` appears twice -- _zval_ptr_dtor and its wrapper -- so the
# anchor is the function signature, which is unique.
ANCHOR = "ZEND_API void _zval_ptr_dtor(zval **zval_ptr ZEND_FILE_LINE_DC)\n{\n"
HEADER = """/* INSTRUMENTED COPY -- diag/patch-zend-execute-api.py. Not for scoring builds. */
extern void ze_dtor_probe(void *p);
"""

def main(src, dst):
    text = open(src, encoding="utf-8", errors="surrogateescape").read()
    if text.count(ANCHOR) != 1:
        sys.exit(f"anchor matched {text.count(ANCHOR)} times, expected 1")
    text = text.replace(ANCHOR, ANCHOR + "\tze_dtor_probe((void *)(*zval_ptr));\n", 1)
    i = text.index("#include ")
    open(dst, "w", encoding="utf-8", errors="surrogateescape").write(text[:i] + HEADER + text[i:])
    print(f"patched _zval_ptr_dtor in {dst}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
