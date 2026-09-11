/* speedtest1 as a Capstone domain.
 *
 * ITS NEAR-TWIN IS speedtest1_measure.c, AND THEY ARE DIFFERENT TOOLS (I-7). This file is the
 * BRING-UP instrument: built by build-sqlite-capstone.sh into speedtest1_capstone.dom, driven by
 * run-sqlite-speedtest1.sh, testset baked in at COMPILE time (SPEEDTEST1_ARGS), ten staged
 * markers so a run that stops somewhere says where, and no cycle counting. The twin is the
 * MEASUREMENT harness: build-sqlite-silicon.sh -> sqlite_silicon.dom, run-speedtest1-measure.sh,
 * testset at RUN time, emits the numbers. Do not reach for one expecting the other's output.
 *
 * SQLite's own benchmark, its source unchanged, with its stdio on the hostcall payload and
 * no files: printf and fprintf render through sqlite3_vsnprintf into the payload the host
 * prints after the domain returns, fopen refuses, unlink is a no-op, and exit writes its
 * code and parks the domain in abort(). The benchmark's main is included under another
 * name and called with fixed arguments (SPEEDTEST1_ARGS).
 *
 * memsys5 is configured here, before the benchmark's own sqlite3_initialize, from the
 * in-image sqlite_heap[] the other SQLite domains use. With SQLITE_LOOKASIDE=1200,40 in the
 * build the allocator chain is the one the paper measures: lookaside above memsys5,
 * memsys5 above nothing.
 *
 * Built by run-sqlite-speedtest1.sh: DOMAIN_SRC=this file, -DSPEEDTEST1_SRC='"<path>"'.
 */
#include <stdint.h>
#include "sqlite3.h"
#include "sqlite_hostcall.h"

#ifndef SQLITE_HEAP_SIZE
#define SQLITE_HEAP_SIZE (256U * 1024U)
#endif
#ifndef SPEEDTEST1_ARGS
#define SPEEDTEST1_ARGS "--memdb", "--size", "1", "--testset", "main", "--verify", "--stats"
#endif
#define CAPSTONE_DPI_REGION_SHARE 1U
/* Bisection aid, see domain_main: -DSPEEDTEST1_STOP_AT=n stops at milestone n. */
#ifndef SPEEDTEST1_STOP_AT
#define SPEEDTEST1_STOP_AT 0
#endif
#ifndef SPEEDTEST1_STACK_ARENA
#define SPEEDTEST1_STACK_ARENA 0
#endif
/* -DSPEEDTEST1_HOOK=1: an instrument linked in beside this file (SPEEDTEST1_HOOK_SRC), called
   from the copies of the sources SQLITE_HOOK_PATCH patched. It defines the three functions
   below; its memsys5 table sits right above the arena in the stack region. */
#ifdef SPEEDTEST1_HOOK
size_t speedtest1_hook_table_bytes(size_t arena_size);
void speedtest1_hook_install(void *arena, size_t arena_size, void *table);
void speedtest1_hook_report(void);
#endif
/* -DSPEEDTEST1_SUBLET=1: the Sublet port of both allocators (sublet/sublet-3530300.patch). The
   pool is the third region the host shares, linear (sqlite_host.user --arena <bytes>); the
   tables memsys5 keeps beside it come from the stack region, where the arena used to be. */
#ifdef SPEEDTEST1_SUBLET
void sqlite3_sublet_grant(void *pLinear);
void sqlite3_sublet_pool(unsigned long *pBase, unsigned long *pEnd);
void sqlite3_sublet_stats(unsigned long *aOut);
#endif
#define CAPSTONE_DELIN(value)                                                \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(value))

static unsigned char sqlite_heap[SQLITE_HEAP_SIZE] __attribute__((aligned(16)));
/* -DSPEEDTEST1_PAD=<bytes>: grow the image by that much and nothing else, to tell a layout effect
   from a code change (bisection aid, 2026-09-10). */
#ifdef SPEEDTEST1_PAD
static unsigned char speedtest1_pad[SPEEDTEST1_PAD] __attribute__((aligned(16), used));
#endif
static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;
/* the host's memory for the allocators: region 2 is the pool (memsys5's heap, or under the
   port the linear grant, which goes to memsys5's slot and not here), region 3 the tables */
static volatile char *pool_region;
static void *tables_region;

/* base and end of a capability the domain may keep in a C variable (a non-linear one) */
static void cap_bounds(void *cap, unsigned long *base, unsigned long *end) {
  unsigned long b, e;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x3" : "=r"(b) : "r"(cap));
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x4" : "=r"(e) : "r"(cap));
  *base = b;
  *end = e;
}

static void output_text(const char *text) {
  if (!hostcall_metadata || !hostcall_payload)
    return;
  /* Both delins guarded for the same reason as sqlite_capstone_domain.c: on the
     gp-captable ABI these capabilities are reached through the cap-table and arrive
     NONLIN, and DELIN on a non-linear capability raises UNEXPECTED_CAP_TYPE on the RTL,
     which WEDGES rather than traps (R-5). QEMU's helper_csdelin returns early, hiding it.
     This is the S-02 root cause, proven on silicon 2026-08-09: with the guard the same
     arm returned in 4 s where it had wedged. UNTESTED IN THIS FILE -- this domain has not
     been re-run on the board since the guard was added; it is the same construct and the
     same ABI, but say so rather than imply it was measured here. */
#ifndef CAPSTONE_GP_CAPTABLE_ABI
  CAPSTONE_DELIN(text);
#endif
  char *payload = (char *)hostcall_payload;
#ifndef CAPSTONE_GP_CAPTABLE_ABI
  CAPSTONE_DELIN(payload);
#endif
  unsigned long offset = hostcall_metadata->length;
  while (*text && offset + 1 < SQLITE_HC_REGION_SIZE)
    payload[offset++] = *text++;
  hostcall_metadata->length = offset;
}

static void output_uint(unsigned long value) {
  char digits[24];
  unsigned count = 0;
  do {
    digits[count++] = (char)('0' + value % 10UL);
    value /= 10UL;
  } while (value);
  char text[24];
  unsigned i = 0;
  while (count)
    text[i++] = digits[--count];
  text[i] = 0;
  output_text(text);
}

/* The payload writers for a source linked in beside this file (an instrument, a probe): the
   domain has no stdout of its own. */
void speedtest1_output_text(const char *text) { output_text(text); }
void speedtest1_output_uint(unsigned long value) { output_uint(value); }

/* The benchmark's stdio: two streams, both the payload. */
struct capstone_sqlite_file {
  int fd;
};
static FILE stream_out = {1};
static FILE stream_err = {2};
FILE *stdout = &stream_out;
FILE *stderr = &stream_err;

int vfprintf(FILE *stream, const char *format, va_list ap) {
  char buffer[1024];
  (void)stream;
  sqlite3_vsnprintf((int)sizeof buffer, buffer, format, ap);
  output_text(buffer);
  return (int)strlen(buffer);
}

int fprintf(FILE *stream, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  int n = vfprintf(stream, format, ap);
  va_end(ap);
  return n;
}

int printf(const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  int n = vfprintf(stdout, format, ap);
  va_end(ap);
  return n;
}

int snprintf(char *buffer, size_t size, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  sqlite3_vsnprintf((int)size, buffer, format, ap);
  va_end(ap);
  return (int)strlen(buffer);
}

int sprintf(char *buffer, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  sqlite3_vsnprintf(64, buffer, format, ap); /* memhook's edge labels, 24-byte buffers */
  va_end(ap);
  return (int)strlen(buffer);
}

int fflush(FILE *stream) {
  (void)stream;
  return 0;
}

FILE *fopen(const char *path, const char *mode) {
  (void)path;
  (void)mode;
  return NULL;
}

int fclose(FILE *stream) {
  (void)stream;
  return 0;
}

char *fgets(char *text, int size, FILE *stream) {
  (void)text;
  (void)size;
  (void)stream;
  return NULL;
}

size_t fwrite(const void *data, size_t size, size_t count, FILE *stream) {
  (void)data;
  (void)stream;
  return size * count;
}

int unlink(const char *path) {
  (void)path;
  return 0;
}

int atoi(const char *text) {
  int sign = 1, value = 0;
  while (*text == ' ' || *text == '\t')
    text++;
  if (*text == '-' || *text == '+')
    sign = *text++ == '-' ? -1 : 1;
  while (*text >= '0' && *text <= '9')
    value = value * 10 + (*text++ - '0');
  return sign * value;
}

/* Return from anywhere. The real entry point is the assembly stub below: it records the frame
   start.S handed over, the stack and the return capability, then continues in
   speedtest1_domain_main. exit() writes its code into the result slot, restores that frame
   and returns to start.S as if domain_main had returned. Without this a fatal error parks the
   domain in abort() forever, the host never regains the core, and the message in the payload
   is never read (measured 2026-09-09: no host heartbeat for 150 s while a domain spun). */
unsigned char speedtest1_exit_frame[32] __attribute__((aligned(16), used));
static unsigned *domain_result;

__asm__(
    "  .text\n"
    "  .globl domain_main\n"
    "domain_main:\n"
    "1: auipc t0, %pcrel_hi(speedtest1_exit_frame)\n"
    "  addi t0, t0, %pcrel_lo(1b)\n"
    "  .insn r 0x5b, 0x1, 0xc, t0, gp, t0\n" /* cincoffset t0, gp, t0: the frame record */
    "  .insn s 0x5b, 0x4, sp, 0(t0)\n"       /* stc sp, 0(t0) */
    "  .insn s 0x5b, 0x4, ra, 16(t0)\n"      /* stc ra, 16(t0) */
    "  j speedtest1_domain_main\n");

__attribute__((noreturn)) void exit(int code) {
  output_text("__CAPSTONE_SPEEDTEST1_EXIT__ code=");
  output_uint((unsigned long)(unsigned)code);
  output_text("\n");
  if (domain_result)
    *domain_result = 0x5117E100u | ((unsigned)code & 0xFFu);
  __asm__ volatile(
      "1: auipc t0, %%pcrel_hi(speedtest1_exit_frame)\n"
      "  addi t0, t0, %%pcrel_lo(1b)\n"
      "  .insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
      "  .insn i 0x5b, 0x3, sp, 0(t0)\n" /* ldc sp, 0(t0) */
      "  .insn i 0x5b, 0x3, ra, 16(t0)\n" /* ldc ra, 16(t0) */
      "  ret\n" ::: "memory");
  for (;;)
    ;
}

/* sqlite3.h maps double to sqlite3_int64 inside its own prototypes when floating point is
   omitted and restores the word at its end; the benchmark's own doubles must follow the same
   mapping or every call that passes one fails to type. Test 300, the Mandelbrot set, then
   computes with integers: the build has no floating point, and that test is not the
   measurement. */
#ifdef SQLITE_OMIT_FLOATING_POINT
#define double sqlite3_int64
#endif
/* Milestones on the payload, so a run that never returns still says how far it got: the
   benchmark's own sqlite3_initialize and sqlite3_open_v2 calls go through these shims. */
static int speedtest1_initialize(void) {
  output_text("__CAPSTONE_SPEEDTEST1_INIT__\n");
  int rc = sqlite3_initialize();
  output_text(rc == SQLITE_OK ? "__CAPSTONE_SPEEDTEST1_INITIALIZED__\n"
                              : "__CAPSTONE_SPEEDTEST1_INIT_FAILED__\n");
  if (SPEEDTEST1_STOP_AT == 3) abort();
  return rc;
}
static int speedtest1_open_v2(const char *name, sqlite3 **db, int flags, const char *vfs) {
  output_text("__CAPSTONE_SPEEDTEST1_OPEN__\n");
  int rc = sqlite3_open_v2(name, db, flags, vfs);
  output_text(rc == SQLITE_OK ? "__CAPSTONE_SPEEDTEST1_OPENED__\n"
                              : "__CAPSTONE_SPEEDTEST1_OPEN_FAILED__\n");
  if (SPEEDTEST1_STOP_AT == 4) abort();
  return rc;
}
#define sqlite3_initialize speedtest1_initialize
#define sqlite3_open_v2 speedtest1_open_v2
#define main speedtest1_main
#include SPEEDTEST1_SRC
#undef main
#undef sqlite3_open_v2
#undef sqlite3_initialize
#ifdef SQLITE_OMIT_FLOATING_POINT
#undef double
#endif

/* -DSPEEDTEST1_PROBE=n: a probe runs instead of the benchmark, as a matched pair with the
   unprotected build. The probe is a source the runner links in beside this file
   (SPEEDTEST1_PROBE_SRC). It returns only when the read it makes did not trap. */
#ifdef SPEEDTEST1_PROBE
void speedtest1_probe(unsigned *res);
#endif

void speedtest1_domain_main(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (shared_region_count == 0)
      hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)res;
    else if (shared_region_count == 1)
      hostcall_payload = (volatile char *)res;
#ifdef SPEEDTEST1_SUBLET
    else if (shared_region_count == 2)
      sqlite3_sublet_grant((void *)res); /* the pool, linear, into its slot and nowhere else */
#else
    else if (shared_region_count == 2)
      pool_region = (volatile char *)res;
#endif
    else if (shared_region_count == 3)
      tables_region = (void *)res;
    ++shared_region_count;
    return;
  }
  /* Bisection aid: -DSPEEDTEST1_STOP_AT=n returns to the host at milestone n with a
     distinctive result the host prints, so a run that dies silently can be narrowed to the
     segment between two milestones. 1 after the entry marker, 2 after memsys5 is configured,
     3 after sqlite3_initialize, 4 after sqlite3_open_v2. */
#define SPEEDTEST1_STOP(n, res) \
  do { if (SPEEDTEST1_STOP_AT == (n)) { *(res) = 0x5117E000u | (n); return; } } while (0)
  if (!hostcall_metadata || !hostcall_payload) {
    *res = SQLITE_HC_ERR_REGION_MISMATCH; /* the shares never arrived: no payload to write */
    return;
  }
  hostcall_metadata->length = 0;
  domain_result = res;
  output_text("__CAPSTONE_SPEEDTEST1_ENTER__\n");
  if (SPEEDTEST1_STOP_AT == 1) {
    /* Report the stack capability's bounds: what dom_data looks like from inside, for carving
       the arena out of it. LCC selector 1 is the total type query (7 = not a capability);
       2, 3, 4 are cursor, start, end and raise on a non-capability, so the type goes first. */
    void *frame = __builtin_frame_address(0);
    unsigned long ty, cur, st, en;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(ty) : "r"(frame));
    output_text("stack cap type="); output_uint(ty);
    if (ty != 7UL) {
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x2" : "=r"(cur) : "r"(frame));
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x3" : "=r"(st) : "r"(frame));
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x4" : "=r"(en) : "r"(frame));
      output_text(" cursor="); output_uint(cur);
      output_text(" start="); output_uint(st);
      output_text(" end="); output_uint(en);
      output_text(" size="); output_uint(en - st);
      output_text(" below_frame="); output_uint(cur - st);
    }
    output_text("\n");
  }
  SPEEDTEST1_STOP(1, res);
  /* Where memsys5's memory comes from. With the host's regions (sqlite_host.user --pool or
     --arena, and --tables), the pool is region 2 and the tables region 3: memsys5's own tables
     under the port, then the instrument's table, and the stack region keeps only the stack.
     Without them, the old carve from the stack region's low end (-DSPEEDTEST1_STACK_ARENA):
     the image itself is capped by the kernel module's order-10 block (Q-01), and the stack
     region is 2.9 MB with the frame at its top (measured 2026-09-09). */
  unsigned char *heap = sqlite_heap;
  int heap_size = (int)sizeof(sqlite_heap);
  unsigned long arena_addr = 0, arena_size = 0;
  unsigned char *hook_table = 0;
  unsigned long hook_room = 0;
  if (shared_region_count >= 4) {
    unsigned long tables_base, tables_end, tables_size;
    if (!tables_region) {
      output_text("__CAPSTONE_SPEEDTEST1_EXIT__ no tables region\n");
      *res = 0x5117E106u;
      return;
    }
    cap_bounds(tables_region, &tables_base, &tables_end);
    tables_size = tables_end - tables_base;
#ifdef SPEEDTEST1_SUBLET
    {
      unsigned long pool_end, atoms;
      sqlite3_sublet_pool(&arena_addr, &pool_end);
      if (pool_end <= arena_addr) {
        output_text("__CAPSTONE_SPEEDTEST1_EXIT__ no pool: the host shared no linear region\n");
        *res = 0x5117E106u;
        return;
      }
      arena_size = pool_end - arena_addr;
      /* memsys5's tables beside the pool: a control byte, a link, a capability per atom and
         the handles of the split blocks, 41 bytes an atom (mem5.c under the port), aligned */
      atoms = arena_size / 64;
      heap_size = (int)(((atoms + 15) & ~15UL) + atoms * 8 + atoms * 16 + (atoms + 32) * 16 + 64);
      heap = tables_region;
      output_text("regions: pool ");
      output_uint(arena_size);
      output_text(" bytes at ");
      output_uint(arena_addr);
      output_text(", linear, tables ");
      output_uint(tables_size);
      output_text(" bytes, memsys5's tables ");
      output_uint((unsigned long)heap_size);
      output_text("\n");
    }
#else
    if (!pool_region) {
      output_text("__CAPSTONE_SPEEDTEST1_EXIT__ no pool region\n");
      *res = 0x5117E106u;
      return;
    }
    {
      unsigned long pool_end;
      cap_bounds((void *)pool_region, &arena_addr, &pool_end);
      arena_size = pool_end - arena_addr;
      heap = (unsigned char *)pool_region;
      heap_size = (int)arena_size;
      output_text("regions: pool ");
      output_uint(arena_size);
      output_text(" bytes at ");
      output_uint(arena_addr);
      output_text(", tables ");
      output_uint(tables_size);
      output_text(" bytes\n");
    }
#endif
    hook_table = (unsigned char *)tables_region + (heap == tables_region ? heap_size : 0);
    hook_room = tables_size - (heap == tables_region ? (unsigned long)heap_size : 0UL);
    if ((unsigned long)heap_size > tables_size && heap == tables_region) {
      output_text("__CAPSTONE_SPEEDTEST1_EXIT__ the tables region is too small for memsys5's tables\n");
      *res = 0x5117E107u;
      return;
    }
  }
#if SPEEDTEST1_STACK_ARENA > 0
  else {
    unsigned char *frame = __builtin_frame_address(0);
    unsigned long cur, st, en;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x2" : "=r"(cur) : "r"(frame));
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x3" : "=r"(st) : "r"(frame));
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x4" : "=r"(en) : "r"(frame));
    heap = frame - (cur - st);
    heap_size = SPEEDTEST1_STACK_ARENA;
    arena_addr = (unsigned long)(uintptr_t)heap;
    arena_size = (unsigned long)heap_size;
#ifdef SPEEDTEST1_SUBLET
    {
      unsigned long pool_end, atoms;
      sqlite3_sublet_pool(&arena_addr, &pool_end);
      if (pool_end <= arena_addr) {
        output_text("__CAPSTONE_SPEEDTEST1_EXIT__ no pool: the host shared no third region\n");
        *res = 0x5117E106u;
        return;
      }
      arena_size = pool_end - arena_addr;
      atoms = arena_size / 64;
      heap_size = (int)(((atoms + 15) & ~15UL) + atoms * 8 + atoms * 16 + (atoms + 32) * 16 + 64);
      output_text("sublet: pool ");
      output_uint(arena_size);
      output_text(" bytes at ");
      output_uint(arena_addr);
      output_text(", the host's region, tables ");
      output_uint((unsigned long)heap_size);
      output_text(" bytes from the stack region's low end, region ");
      output_uint(en - st);
      output_text(" bytes, stack keeps ");
      output_uint((en - st) - (unsigned long)heap_size);
      output_text("\n");
    }
#else
    output_text("arena: ");
    output_uint((unsigned long)heap_size);
    output_text(" bytes from the stack region's low end, region ");
    output_uint(en - st);
    output_text(" bytes, stack keeps ");
    output_uint((en - st) - (unsigned long)heap_size);
    output_text("\n");
#endif
    hook_table = heap + heap_size;
    hook_room = (en - st) - (unsigned long)heap_size - 393216UL; /* the stack keeps 384 KB */
  }
#endif
#if defined(SPEEDTEST1_HOOK) && !defined(SPEEDTEST1_HOOK_NO_INSTALL)
  if (hook_table) {
    unsigned long table_bytes = speedtest1_hook_table_bytes((size_t)arena_size);
    output_text("hook table: ");
    output_uint(table_bytes);
    output_text(" bytes, room ");
    output_uint(hook_room);
    output_text("\n");
    if (table_bytes > hook_room) {
      output_text("__CAPSTONE_SPEEDTEST1_EXIT__ no room for the instrument's table\n");
      *res = 0x5117E104u;
      return;
    }
    speedtest1_hook_install((void *)(uintptr_t)arena_addr, (size_t)arena_size, hook_table);
  }
#endif
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, heap, heap_size, 64);
  if (rc != SQLITE_OK) {
    output_text("__CAPSTONE_SPEEDTEST1_EXIT__ config-heap\n");
    *res = SQLITE_HC_ERR_CONFIG_HEAP;
    return;
  }
  output_text("__CAPSTONE_SPEEDTEST1_START__\n");
  SPEEDTEST1_STOP(2, res);
#ifdef SPEEDTEST1_PROBE
  speedtest1_probe(res); /* returns only when the read it makes did not trap */
  return;
#endif
  static char *argv[] = {"speedtest1", SPEEDTEST1_ARGS, 0};
  int argc = (int)(sizeof(argv) / sizeof(argv[0])) - 1;
  rc = speedtest1_main(argc, argv);
#ifdef SPEEDTEST1_HOOK
  speedtest1_hook_report();
#endif
#ifdef SPEEDTEST1_SUBLET
  {
    unsigned long v[5];
    sqlite3_sublet_stats(v);
    output_text("sublet: split=");
    output_uint(v[0]);
    output_text(" mrev=");
    output_uint(v[1]);
    output_text(" delin=");
    output_uint(v[2]);
    output_text(" revoke=");
    output_uint(v[3]);
    output_text(" init=");
    output_uint(v[4]);
    output_text("\n");
  }
#endif
  output_text("__CAPSTONE_SPEEDTEST1_DONE__ rc=");
  output_uint((unsigned long)(unsigned)rc);
  output_text("\n");
  *res = SQLITE_HC_RET_DONE;
}
