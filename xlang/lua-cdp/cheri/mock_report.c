/* CHERI-side reporter for the Lua-CDP shims.
 *
 * printf is forward-declared rather than pulled from <stdio.h>: this CHERI SDK
 * clang does not redirect the <...> header search into the purecap sysroot
 * (it lists the host /usr/include), so <stdio.h> would grab host glibc headers
 * that assume 8-byte pointers. The link still uses the sysroot's libc/crt, which
 * is a separate path and works. The shims themselves need no sysroot headers
 * (only <stdint.h>, a clang builtin), so this file is the only one affected.
 *
 * The shim prints this line ONLY on the MISS path (stale access completed). A
 * caught access dies by capability fault (signal) before reaching it; classify.py
 * reads that from the exit code (rc >= 128), not from this marker. */
int printf(const char *fmt, ...);

void mock_report(const char *row, const char *what) {
  printf("MOCK %s %s\n", row, what);
}
