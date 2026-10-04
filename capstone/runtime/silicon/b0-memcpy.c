/* B0.8 (docs/plans/b0-silicon-delegated-runtime.md): does the runtime's guarded memcpy survive R-29 on silicon?
 *
 * ISSUES R-29: a 128-bit load of a granule returns a WRONG HIGH HALF while a plain store to that granule is still in
 * the write buffer. It reads 0 after a fresh low-word store (or both words), and the OLD value after a fresh
 * high-word store. The runtime's memcpy (string_bounds_safe.c) copies aligned granules with that load. Built with
 * CAPSTONE_MEMCPY_PLAIN_GUARD it asks LCC's type query first and copies plain data with 8-byte moves.
 *
 * A matched pair, differing in ONE thing:
 * - copy_unguarded is the runtime's memcpy loop without the guard, the same shape as before the change;
 * - memcpy is the guarded one.
 * Both are called the same way, after the same fresh stores, with six stores to another line queued ahead of them so
 * the pair stays in the write buffer (the RTL lane's "busy" arm). The unguarded arm is the POSITIVE CONTROL: on
 * silicon it must miscopy, or the run created no hazard and says nothing about the guard. QEMU has no R-29, so there
 * the control cannot fire and the run reports exit 2 (void) by design.
 *
 * Exit status: 0 = the control miscopied and the guarded copy never did; 1 = the guarded copy miscopied (whatever
 * the control did); 2 = the control never miscopied, so no hazard was created and the run is void. One line per
 * face is printed either way. */
#include <stdio.h>
#include <string.h>

#define REPS 32
typedef unsigned long u64;
typedef void *cap_t;

static void __attribute__((noinline)) copy_unguarded(void *dst, const void *src, size_t n)
{
	unsigned char *d = dst;
	const unsigned char *s = src;
	if (((u64)d & 15) == ((u64)s & 15)) {
		while (n && ((u64)d & 15)) { *d++ = *s++; n--; }
		while (n >= 16) {
			*(cap_t *)d = *(const cap_t *)s;
			d += 16; s += 16; n -= 16;
		}
	}
	while (n--) *d++ = *s++;
}

static unsigned char granule[16] __attribute__((aligned(16)));
static unsigned char busy[64] __attribute__((aligned(64)));
static unsigned char out[16] __attribute__((aligned(16)));

/* No printf: vfprintf is not in the gp-captable musl archive (its long-double formatting needs the fp128 libcalls
   that C-43 drops), so lines are built here and written with fputs. */
static char *put_str(char *p, const char *s) { while (*s) *p++ = *s++; return p; }
static char *put_uint(char *p, unsigned v)
{
	char t[12];
	int n = 0;
	do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v);
	while (n) *p++ = t[--n];
	return p;
}

static void fence(void) { __asm__ volatile ("fence" ::: "memory"); }
static void store64(void *p, u64 v) { *(volatile u64 *)p = v; }
static u64 load64(const void *p) { return *(const volatile u64 *)p; }

/* face: 0 = fresh low word only, 1 = fresh high word only, 2 = both. The old contents are written and drained
   first, so "old" is in memory and only the face's stores are in the write buffer at the copy. */
static int one(int face, int guarded, u64 seed)
{
	u64 old_lo = 0x1111000000000000UL | seed, old_hi = 0x2222000000000000UL | seed;
	u64 new_lo = 0x3333000000000000UL | seed, new_hi = 0x4444000000000000UL | seed;
	store64(granule, old_lo);
	store64(granule + 8, old_hi);
	store64(out, 0);
	store64(out + 8, 0);
	fence();
	for (int i = 0; i < 6; i++) store64(busy + 8 * i, seed + i);   /* queued ahead of the pair */
	if (face != 1) store64(granule, new_lo);
	if (face != 0) store64(granule + 8, new_hi);
	if (guarded) memcpy(out, granule, 16);
	else copy_unguarded(out, granule, 16);
	fence();
	u64 want_lo = face != 1 ? new_lo : old_lo, want_hi = face != 0 ? new_hi : old_hi;
	return load64(out) != want_lo || load64(out + 8) != want_hi;
}

int main(void)
{
	static const char *const faces[3] = {"low", "high", "both"};
	int control = 0, guarded = 0;
	for (int face = 0; face < 3; face++) {
		int bad[2] = {0, 0};
		for (int r = 0; r < REPS; r++)
			for (int g = 0; g < 2; g++)
				bad[g] += one(face, g, (u64)(face * 1000 + r * 2 + g + 1));
		char line[96], *p = line;
		p = put_str(p, "B0.8 face="); p = put_str(p, faces[face]);
		p = put_str(p, " unguarded_bad="); p = put_uint(p, (unsigned)bad[0]);
		p = put_str(p, " guarded_bad="); p = put_uint(p, (unsigned)bad[1]);
		p = put_str(p, " of "); p = put_uint(p, REPS); p = put_str(p, "\n"); *p = 0;
		fputs(line, stdout);
		control += bad[0];
		guarded += bad[1];
	}
	char line[80], *p = line;
	p = put_str(p, "B0.8 total unguarded_bad="); p = put_uint(p, (unsigned)control);
	p = put_str(p, " guarded_bad="); p = put_uint(p, (unsigned)guarded); p = put_str(p, "\n"); *p = 0;
	fputs(line, stdout);
	fflush(stdout);
	if (guarded) return 1;
	return control ? 0 : 2;
}
