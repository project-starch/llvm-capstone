#ifndef CAPSTONE_REPRESENTABLE_H
#define CAPSTONE_REPRESENTABLE_H
/* Is a [base, base+len) range exactly representable in the 27-bit compressed bounds format?
 *
 * BOTH EDGES MATTER, and they fail differently. The encoder
 * (capstone-ariane core/include/ariane_pkg.sv, compress_bounds) rounds the END up and
 * TRUNCATES THE BASE DOWN:
 *
 *     B[13:3] = {bounds.start >> E}[13:3];                 // base: truncated, no compensation
 *     T[11:3] = {bounds.base  >> E}[11:3];
 *     if (((bounds.base >> (E+3)) << (E+3)) != bounds.base)
 *         T[11:3] += 1;                                    // end: rounded up
 *
 * so an unaligned base silently gains authority BELOW the object, and the kernel states the rule
 * directly: "A region is representable only if BOTH its length and its base are granule multiples"
 * (modcapstone capstone.c:338, enforced at :344 and process.c:124).
 *
 * THE R-54 RENORMALISATION CANNOT FIRE ON A RANGE THIS PREDICATE ACCEPTS, which is why the
 * granule below is the plain 2^(msb(len)-9) and not a worst case. The renormalisation is
 * `if ((tg - bg) >= 64'd1024) E = E + 1;` with tg = roundup(end, g)/g and bg = floor(start/g)
 * (ariane_pkg.sv). When the base and the length are both granule multiples, tg - bg is exactly
 * len/g, and len < 2^(msb(len)+1) with g = 2^(msb(len)-9) gives len/g < 1024 -- so the condition
 * is never met. It fires only for ranges that are already unrepresentable, which this predicate
 * rejects for the simpler reason. An earlier version doubled the granule inside a supposed
 * renormalisation band and WRONGLY REJECTED exactly representable ranges (len = 16368 at an
 * aligned base is one); the cross-check below caught it.
 *
 * Verified against the project's bounds encoder oracle over the full small range and random large
 * values: capstone/tests/exact-bounds-iii/repr-cross-check.py, which also negative-tests the
 * predicate so a wrong answer fails rather than passing quietly.
 */

/* The granule for a length, as the worst case over every base: 1 below 4 KiB (the encoder's exact
 * branch), else 2^(msb(len)-9), doubled inside the renormalisation band -- the top granule-width of
 * the length's own exponent range. */
static inline unsigned long capstone_repr_granule(unsigned long len)
{
	unsigned k, e;
	unsigned long g;
	if (len < 4096UL)
		return 1UL;
	k = 63u - (unsigned)__builtin_clzl(len);   /* msb(len) */
	e = k - 12u;
	g = 1UL << (e + 3u);                       /* 2^(msb(len)-9) */
	return g;
}

/* A length rounded up so that it is a granule multiple. */
static inline unsigned long capstone_repr_round(unsigned long len)
{
	unsigned long g = capstone_repr_granule(len ? len : 1UL);
	unsigned long r = (((len ? len : 1UL) + g - 1UL) / g) * g;
	/* Rounding can cross into a coarser granule; one re-check settles it, and the loop is
	 * bounded because each step at most doubles the granule. */
	while (r % capstone_repr_granule(r))
		r = ((r + capstone_repr_granule(r) - 1UL) / capstone_repr_granule(r))
		    * capstone_repr_granule(r);
	return r;
}

/* Is [base, base+len) exactly representable? Both edges must be granule multiples. */
static inline int capstone_repr_ok(unsigned long base, unsigned long len)
{
	unsigned long g;
	if (!len)
		return 0;
	g = capstone_repr_granule(len);
	return (base % g) == 0UL && (len % g) == 0UL;
}

/* The base a caller must place an object at, at or below `base`, for `len` to be representable. */
static inline unsigned long capstone_repr_base_down(unsigned long base, unsigned long len)
{
	unsigned long g = capstone_repr_granule(len ? len : 1UL);
	return base - (base % g);
}

#endif /* CAPSTONE_REPRESENTABLE_H */
