/* ---------------- LadyBug reachability probe: implementation ----------------
 *
 * Injected into the amalgamation directly after memsys5's `mem5` globals, because
 * the whole point is to answer a question only the allocator can answer: at the
 * instant the defect site touches a pointer, is the block that pointer came from
 * CURRENTLY FREE, and is the access inside the bytes that were actually requested?
 *
 * CHERI cannot answer either question under memsys5. memsys5 hands out
 * &mem5.zPool[i*szAtom], so every sub-allocation inherits the capability bounds of
 * the whole 256 KiB arena: a use-after-free is in bounds, and so is an overflow
 * that stays in the arena. That is the measurement this corpus exists to make --
 * but it is only a measurement once we can show the defective access HAPPENED.
 *
 * The shadow map is ours, not memsys5's. Reading memsys5's own aCtrl[] would mean
 * walking backwards to find the block start (aCtrl is only meaningful at the first
 * atom of a block, and memsys5 leaves stale bytes in the interior), which is both
 * O(nBlock) and ambiguous. Recording ownership on the way in is O(1) at the probe
 * and -- the part that matters -- it keeps the REQUESTED size, so an overflow that
 * stays inside the rounded-up block is still reported. memsys5 rounds 60 bytes up
 * to a 64-byte atom; without the requested size, a 3-byte overrun is invisible.
 *
 * The probe never allocates and never calls stdio: it writes to fd 2 with write(2)
 * and formats by hand. A probe that mallocs would perturb the allocator it is
 * measuring, and a buffered probe would lose its last line to the fault it is
 * there to witness.
 */
#ifndef LB_PROBE_IMPL_INCLUDED
#define LB_PROBE_IMPL_INCLUDED

#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>

#define LB_MAX_ATOMS 65536   /* 4 MiB at a 64-byte atom; see lb_shadow_off below */

int lb_site = 0;
static int   lb_shadow_off = 0;     /* 1 = arena too large to shadow; say so, loudly */
static unsigned lb_owner[LB_MAX_ATOMS];  /* atom -> owning block's first atom, +1 */
static unsigned lb_req[LB_MAX_ATOMS];    /* at a block's first atom: bytes REQUESTED */
static unsigned char lb_freed[LB_MAX_ATOMS]; /* at a block's first atom: 1 = freed */
static unsigned long lb_hits = 0, lb_bad = 0, lb_first_bad = 0;

/* ---- hand-rolled output. No stdio: see the header comment. ---- */
static void lb_puts(const char *s){ const char *e=s; while(*e) e++; (void)write(2, s, (size_t)(e-s)); }
static void lb_putu(unsigned long v){
  char b[24]; int i=24; if(!v){ lb_puts("0"); return; }
  while(v){ b[--i] = (char)('0' + (v%10)); v/=10; }
  (void)write(2, b+i, (size_t)(24-i));
}
static void lb_puti(long v){ if(v<0){ lb_puts("-"); lb_putu((unsigned long)(-v)); } else lb_putu((unsigned long)v); }
static void lb_putx(unsigned long long v){
  char b[20]; int i=20; const char *h="0123456789abcdef";
  lb_puts("0x"); if(!v){ lb_puts("0"); return; }
  while(v){ b[--i] = h[v & 15]; v >>= 4; }
  (void)write(2, b+i, (size_t)(20-i));
}

/* ---- what the hardware thinks of this pointer ----
 * Printed for EVERY witness, on both allocator arms, because it is the headline
 * finding and not a footnote: tag=1 with the access inside [base,base+len) is
 * precisely the state in which CHERI is correct to stay silent. */
static void lb_put_cap(const void *p){
#ifdef __CHERI_PURE_CAPABILITY__
  lb_puts(" tag="); lb_putu((unsigned long)__builtin_cheri_tag_get(p));
  lb_puts(" cb="); lb_putx((unsigned long long)__builtin_cheri_base_get(p));
  lb_puts("+"); lb_putu((unsigned long)__builtin_cheri_length_get(p));
  lb_puts(" ca="); lb_putx((unsigned long long)__builtin_cheri_address_get(p));
#else
  lb_puts(" cap=n/a ca="); lb_putx((unsigned long long)(uintptr_t)p);
#endif
}

/* ---- shadow map maintenance, called from memsys5's public wrappers ---- */
static int lb_atom_of(const void *p, int *pOk){
  long off;
  *pOk = 0;
  if( mem5.zPool==0 || mem5.szAtom<=0 ) return 0;
  off = (long)((const unsigned char*)p - (const unsigned char*)mem5.zPool);
  if( off < 0 || off >= (long)mem5.nBlock * mem5.szAtom ) return 0;
  *pOk = 1;
  return (int)(off / mem5.szAtom);
}
static void lb_on_alloc(void *p, int nByte){
  int ok, i, n, k;
  if( lb_shadow_off || p==0 ) return;
  i = lb_atom_of(p, &ok); if( !ok ) return;
  if( mem5.nBlock > LB_MAX_ATOMS ){
    lb_shadow_off = 1;
    lb_puts("LB_PROBE DISABLED arena has "); lb_putu((unsigned long)mem5.nBlock);
    lb_puts(" atoms > LB_MAX_ATOMS; raise it and rebuild\n");
    return;
  }
  /* memsys5 gives back a power-of-two run of atoms; mark every atom of it so an
   * interior pointer (which is what a dangling field pointer usually is) resolves. */
  n = 1; k = (int)mem5.aCtrl[i] & 0x1f; while( k-- > 0 ) n <<= 1;
  if( i + n > LB_MAX_ATOMS ) n = LB_MAX_ATOMS - i;
  for( k = 0; k < n; k++ ) lb_owner[i+k] = (unsigned)i + 1u;
  lb_req[i]   = (unsigned)(nByte > 0 ? nByte : 0);
  lb_freed[i] = 0;
}
static void lb_on_free(void *p){
  int ok, i;
  if( lb_shadow_off || p==0 ) return;
  i = lb_atom_of(p, &ok); if( !ok ) return;
  if( i < LB_MAX_ATOMS ) lb_freed[i] = 1;
}

/* ---- the probe proper ---- */
int lb_hit_slow(int id, const void *p, long n){
  int ok, i, b;
  lb_hits++;
  if( p==0 ){
    lb_bad++; if(!lb_first_bad) lb_first_bad = lb_hits;
    lb_puts("LB_WITNESS site="); lb_putu((unsigned long)id);
    lb_puts(" kind=NULL n="); lb_puti(n); lb_puts("\n");
    return 1;
  }
  if( n < 0 ){   /* 2c7a73eaea_0 reaches memcpy with a negative length */
    lb_bad++; if(!lb_first_bad) lb_first_bad = lb_hits;
    lb_puts("LB_WITNESS site="); lb_putu((unsigned long)id);
    lb_puts(" kind=NEGSIZE n="); lb_puti(n); lb_put_cap(p); lb_puts("\n");
    return 1;
  }
  i = lb_atom_of(p, &ok);
  if( !ok || lb_shadow_off ){
    /* Not in the memsys5 arena: either the _sys arm (system malloc) or a static /
     * stack pointer. On purecap the capability bounds ARE the object bounds there,
     * so spatial is still answerable; temporal is not, and is reported as such
     * rather than quietly scored OK. */
#ifdef __CHERI_PURE_CAPABILITY__
    {
      unsigned long long base = (unsigned long long)__builtin_cheri_base_get(p);
      unsigned long long len  = (unsigned long long)__builtin_cheri_length_get(p);
      unsigned long long a    = (unsigned long long)__builtin_cheri_address_get(p);
      if( __builtin_cheri_tag_get(p)==0 || a < base || a + (unsigned long long)n > base + len ){
        lb_bad++; if(!lb_first_bad) lb_first_bad = lb_hits;
        lb_puts("LB_WITNESS site="); lb_putu((unsigned long)id);
        lb_puts(" kind=CAP-OOB n="); lb_puti(n); lb_put_cap(p); lb_puts("\n");
      }
    }
#endif
    return 1;
  }
  b = (int)lb_owner[i];
  if( b == 0 ){
    /* Inside the arena but owned by nobody we saw allocated: the block was handed
     * out before the probe armed, or it is arena space memsys5 has never issued. */
    return 1;
  }
  b -= 1;
  if( lb_freed[b] || (long)((const unsigned char*)p - ((const unsigned char*)mem5.zPool + (long)b*mem5.szAtom)) + n > (long)lb_req[b] ){
    const char *kind = lb_freed[b] ? "UAF" : "OOB";
    long off = (long)((const unsigned char*)p - ((const unsigned char*)mem5.zPool + (long)b*mem5.szAtom));
    lb_bad++; if(!lb_first_bad) lb_first_bad = lb_hits;
    lb_puts("LB_WITNESS site="); lb_putu((unsigned long)id);
    lb_puts(" kind="); lb_puts(kind);
    lb_puts(" off="); lb_puti(off);
    lb_puts(" n="); lb_puti(n);
    lb_puts(" req="); lb_putu((unsigned long)lb_req[b]);
    lb_puts(" blk="); lb_putx((unsigned long long)(uintptr_t)((const unsigned char*)mem5.zPool + (long)b*mem5.szAtom));
    lb_put_cap(p);
    lb_puts("\n");
  }
  return 1;
}

/* Not static, and called from the case's SIGPROT handler as well as atexit:
 * the handler ends in _exit(), which does not run atexit, so a run that FAULTS
 * would otherwise carry no hit count -- and the hit count is what says the fault
 * happened at the probed site rather than somewhere else. */
void lb_report(void){
  lb_puts("LB_SUMMARY site="); lb_putu((unsigned long)lb_site);
  lb_puts(" hits="); lb_putu(lb_hits);
  lb_puts(" witnesses="); lb_putu(lb_bad);
  lb_puts(" first_witness_at_hit="); lb_putu(lb_first_bad);
  lb_puts("\n");
}
/* The constructor ALWAYS prints, armed or not. Six times in this project a pattern
 * that could never match was read as "no problem found"; a run whose log has no
 * LB_ARMED line is a run where the probe was not compiled in, and that must be
 * distinguishable from a run where the site was never reached. */
__attribute__((constructor)) static void lb_init(void){
  const char *s = getenv("LB_SITE");
  lb_site = s ? atoi(s) : 0;
  lb_puts("LB_ARMED site="); lb_putu((unsigned long)lb_site); lb_puts("\n");
  atexit(lb_report);
}
#endif /* LB_PROBE_IMPL_INCLUDED */
