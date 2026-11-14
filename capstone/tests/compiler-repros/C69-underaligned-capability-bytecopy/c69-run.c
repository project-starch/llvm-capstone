/* C-69 run-time test: a capability read or written through an address whose 16-byte alignment the compiler cannot
 * prove must keep its tag when the address is aligned at run time, and must give memcpy's bytes, untagged and
 * without a trap, when it is not.
 *
 * The shapes, each in a noinline function so the alignment stays unknown:
 *   load_void   FFmpeg's av_freep: memcpy(&val, arg, sizeof val) with arg a void *; after SROA an align-1
 *               capability load, with no memcpy left (the -fno-builtin build calls musl's instead)
 *   store_void  the store half: memcpy(arg, &v, sizeof v)
 *   load_word   the same through an unsigned long *: align 8
 *   load_field  libavfilter's FF_FIELD_AT: *(void **)((char *)&obj + runtime offset), align 1
 *   store_field its store half
 * Each runs at a 16-aligned address holding a tagged capability (the tag must survive, and the pointer must
 * dereference to its cell) and at a misaligned one (the bytes must equal the runtime memcpy's, called through a
 * volatile pointer so it stays a call, and the tag must be clear). The control is a plain-word copy of an aligned
 * capability, which must lose the tag: it shows the tag read can report a loss. Tags are read with the compiler
 * builtin, never inferred from values; every read-back the checks make is itself 16-aligned in the IR.
 *
 * At -O0 there is no SROA: load_void, load_word and store_void stay memcpy calls (musl's, which keeps the tag), so
 * only the FF_FIELD_AT shapes are C-69 shapes there. Output: C69_CHECK:<name>:PASS|FAIL lines, then C69_DONE:<checks>:<failures>. */
#include <stdio.h>
#include <string.h>

#define TAG(p) __builtin_capstone_cap_get_tag((void *)(p))
#define BARRIER() __asm__ volatile("" ::: "memory")
#define FF_FIELD_AT(type, off, obj) (*(type *)((char *)&(obj) + (off)))

struct cfg { void *formats, *samplerates, *channel_layouts; };
struct link { int pad; struct cfg incfg; };

static int cells[4];
static unsigned char buf[96] __attribute__((aligned(16)));
static unsigned char ref[96] __attribute__((aligned(16)));
static struct link lk;
static int checks, failures;
static void *(*volatile memcpy_call)(void *, const void *, unsigned long) = memcpy;
static volatile unsigned field_off = __builtin_offsetof(struct cfg, samplerates);

static void check(int ok, const char *name)
{
	++checks;
	failures += !ok;
	printf("C69_CHECK:%s:%s\n", name, ok ? "PASS" : "FAIL");
}

__attribute__((noinline)) void *load_void(void *arg)
{
	void *val;
	memcpy(&val, arg, sizeof(val));
	return val;
}
__attribute__((noinline)) void store_void(void *arg, void *v) { memcpy(arg, &v, sizeof(v)); }
__attribute__((noinline)) void *load_word(unsigned long *arg)
{
	void *val;
	memcpy(&val, arg, sizeof(val));
	return val;
}
__attribute__((noinline)) void *load_field(struct link *l, unsigned off) { return FF_FIELD_AT(void *, off, l->incfg); }
__attribute__((noinline)) void store_field(struct link *l, unsigned off, void *v) { FF_FIELD_AT(void *, off, l->incfg) = v; }
/* The control: plain 8-byte words, which can never carry a tag. */
__attribute__((noinline)) void *copy_words(void *arg)
{
	void *val;
	volatile unsigned long *d = (volatile unsigned long *)&val;
	const volatile unsigned long *s = (const volatile unsigned long *)arg;
	d[0] = s[0];
	d[1] = s[1];
	return val;
}

/* Places the capability &cells[i] at buf + off, the way memcpy would: tagged only when off is 16-aligned. */
static void place(unsigned off, int i)
{
	void *p = &cells[i];
	memset(buf, 0, sizeof buf);
	memcpy_call(buf + off, &p, sizeof p);
	BARRIER();
}

/* The runtime's result for reading 16 bytes at buf + off into an aligned object. */
static void *reference_load(unsigned off)
{
	void *val;
	memcpy_call(&val, buf + off, sizeof val);
	return val;
}

static int same_bytes(const void *a, const void *b) { return memcmp(a, b, 16) == 0; }

static void loads(void)
{
	static const unsigned offs[] = {16, 1, 8};
	for (unsigned k = 0; k < 3; ++k) {
		unsigned off = offs[k];
		char nm[64];
		place(off, 1);
		void *want = reference_load(off);
		void *got = load_void(buf + off);
		snprintf(nm, sizeof nm, "load_void_off%u", off);
		if (off % 16 == 0)
			check(TAG(got) == 1 && got == &cells[1] && *(int *)got == 11, nm);
		else
			check(TAG(got) == 0 && TAG(want) == 0 && same_bytes(&got, &want), nm);
		if (off % 8 == 0) {
			got = load_word((unsigned long *)(buf + off));
			snprintf(nm, sizeof nm, "load_word_off%u", off);
			if (off % 16 == 0)
				check(TAG(got) == 1 && got == &cells[1] && *(int *)got == 11, nm);
			else
				check(TAG(got) == 0 && TAG(want) == 0 && same_bytes(&got, &want), nm);
		}
	}
}

static void stores(void)
{
	static const unsigned offs[] = {16, 1};
	for (unsigned k = 0; k < 2; ++k) {
		unsigned off = offs[k];
		char nm[64];
		void *p = &cells[2];
		memset(buf, 0, sizeof buf);
		memset(ref, 0, sizeof ref);
		store_void(buf + off, p);
		memcpy_call(ref + off, &p, sizeof p);
		BARRIER();
		snprintf(nm, sizeof nm, "store_void_off%u", off);
		if (off % 16 == 0) {
			/* The read-back must not itself be an under-aligned capability load (buf + a runtime offset is
			   align 1), or it strips the tag store_void kept: state the alignment the branch has checked. */
			void *back = *(void **)__builtin_assume_aligned(buf + off, 16);
			check(TAG(back) == 1 && back == &cells[2] && *(int *)back == 21 && same_bytes(buf + off, ref + off), nm);
		} else {
			check(same_bytes(buf + off, ref + off), nm);
		}
	}
}

static void fields(void)
{
	void *p = &cells[3];
	lk.incfg.samplerates = 0;
	BARRIER();
	store_field(&lk, field_off, p);
	BARRIER();
	check(TAG(lk.incfg.samplerates) == 1 && lk.incfg.samplerates == &cells[3], "store_field");
	void *got = load_field(&lk, field_off);
	check(TAG(got) == 1 && got == &cells[3] && *(int *)got == 31, "load_field");
}

int main(void)
{
	for (int i = 0; i < 4; ++i)
		cells[i] = 10 * i + 1;
	loads();
	stores();
	fields();
	place(16, 0);
	check(TAG(copy_words(buf + 16)) == 0, "control_word_copy_drops_tag");
	printf("C69_DONE:%d:%d\n", checks, failures);
	return failures != 0;
}
