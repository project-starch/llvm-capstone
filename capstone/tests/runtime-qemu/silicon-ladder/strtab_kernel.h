#ifndef STRTAB_KERNEL_H
#define STRTAB_KERNEL_H
/* S1.5 gate: POINTER-VALUED initializers -- the exact SQLite shape.
 *
 * An array of `const char *` and a table of function pointers both load UNTAGGED
 * from the static image; only .capstone_cap_init turns them into capabilities.
 * SQLite has 54 such globals (sqlite3StdType, the VFS method tables, FuncDef.xSFunc,
 * sqlite3aLTb/aEQb/aGTb). No pre-existing ladder rung has one, which is exactly why
 * the silicon glue never running the cap-init table went unnoticed.
 *
 * STATUS 2026-07-28: KNOWN-FAILING, blocked on a DIFFERENT fix than the one it was
 * written for. Under -capstone-gp-captable a FUNCTION address is not an indexable
 * global, so it falls through lowerGlobalAddress to the LGA path and is materialized
 * as `scc rd, gp, &f` -- but gp is bounded to the CAP TABLE, so the result is a
 * capability with cap-table bounds and a wild cursor. Measured: this rung emits 5
 * `scc` against a passing rung's 2, and the fault is
 *   Cap mem access OOB: rs1 = x9, cursor = 10156051c, size = 4,
 *                       bounds = (10157ff80, 101580000)   <- the cap table
 * i.e. exactly that shape. helper_csscc does not bounds-check, which is why this has
 * been latent: it silently "works" on QEMU for any rung that never dereferences such a
 * pointer, and no ladder rung did.
 *
 * The fix is the deferred S1a item: for a code symbol under the gp-free ABI, hand back
 * the raw PseudoLLA integer with no gp involvement -- what SelectCall already does for
 * direct calls. Until then this rung gates the FUNCTION-POINTER path, not cap-init.
 * The cap-init loop itself is not implicated: rungs with an empty .capstone_cap_init
 * table (beebs_aha_mont64, beebs_crc32big) still pass with RUN_CAP_INIT present. */
static const char *const st_names[6] = {
  "integer", "real", "text", "blob", "null", "capability"
};
static unsigned st_add(unsigned a){ return a + 3u; }
static unsigned st_xor(unsigned a){ return a ^ 0x5a5au; }
static unsigned st_rot(unsigned a){ return (a << 7) | (a >> 25); }
static unsigned (*const st_fns[3])(unsigned) = { st_add, st_xor, st_rot };

static unsigned strtab_compute(void){
  unsigned h = 2166136261u;
  for (int rep = 0; rep < 32; rep++) {
    for (int i = 0; i < 6; i++) {
      const char *s = st_names[i];              /* pointer global -> needs cap-init */
      for (int k = 0; s[k]; k++) { h ^= (unsigned char)s[k]; h *= 16777619u; }
    }
    for (int i = 0; i < 3; i++) h = st_fns[i](h); /* fn-pointer table -> same */
  }
  return h;
}
#endif
