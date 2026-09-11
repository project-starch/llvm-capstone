#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_hostcall.h"

/* MARKERS WITHOUT STDIO, AND SHORT ENOUGH TO ESCAPE.
 *
 * Two independent properties, both forced by the board capture (2026-07-30, three
 * SQLite attempts): after libcapstone's last line the console shows at most
 * "sqlite-host: cre" -- 16 bytes, the 8250 TX FIFO depth -- and then the *bootrom*
 * banner, i.e. the core RESETS. Only what is already sitting in the FIFO ever
 * transmits, so a marker longer than 16 bytes is a marker that cannot be read.
 *
 * 1. write(2), not fprintf(). Removes glibc's vfprintf engine (buffering mode
 *    selection, the unbuffered-stderr stack buffer, __printf_buffer's dispatch) from
 *    the one instrument that has to survive the failure. This is exactly what every
 *    freestanding ladder controller already does successfully on this board
 *    (rtl-smoke/ladder_perf_ctl.c `puts_`), so the marker path becomes identical to a
 *    proven-working one. No varargs, no locale, no buffering, one syscall.
 * 2. Every marker is <= 16 bytes, so one FIFO load carries a whole marker and the
 *    LAST marker on the console names the last phase reached. Values go on their own
 *    following line, so a lost value can never cost a lost phase.
 *
 * Phases: A dom-ok, B before create_region #1, C before #2, D both mapped,
 * E before share #1, F before share #2, G before call_dom, H after call_dom,
 * X the failure path. */
static void mark(const char *s) {
  unsigned long n = 0;
  while (s[n])
    n++;
  (void)write(STDOUT_FILENO, s, n);   /* fd 1: stderr resets the core */
}

/* Decimal, no varargs: hand-rolled so the value-carrying markers keep their meaning
   without reintroducing stdio. Single write, so it cannot tear between digits. */
static void mark_u(const char *prefix, unsigned long v) {
  char buf[24];
  int i = (int)sizeof(buf);
  if (!v)
    buf[--i] = '0';
  while (v) {
    buf[--i] = (char)('0' + (v % 10));
    v /= 10;
  }
  mark(prefix);
  (void)write(STDOUT_FILENO, buf + i, sizeof(buf) - (unsigned long)i);
  mark("\n");
}

static int fail_cleanup(const char *message, unsigned long value) {
  /* X FIRST, on purpose. "sqlite-host: create_dom failed (observed=...)",
     "sqlite-host: create_dom ok (id=...)" and "sqlite-host: create_region #1" all
     share their first 16 characters, so the previous markers could not tell a FAILED
     create_dom from a successful one -- and a failed create_dom means the ioctl
     returned an error, which is a completely different bug from a wedge. */
  mark("SQ: X/fail\n");
  mark_u("SQ: obs=", value);
  mark(message);
  mark("\n");
  capstone_cleanup();
  return 1;
}

static int tail_payload;

struct tail_state {
  volatile struct sqlite_hostcall_v0 *metadata;
  const char *payload;
  unsigned long printed;
  int stop;
};

static void tail_flush(struct tail_state *t) {
  unsigned long length = t->metadata->length;
  if (length > SQLITE_HC_REGION_SIZE)
    return;
  if (length > t->printed) {
    (void)write(STDOUT_FILENO, t->payload + t->printed, (size_t)(length - t->printed));
    t->printed = length;
  }
}

static void *tail_main(void *arg) {
  struct tail_state *t = arg;
  unsigned ticks = 0;
  while (!__atomic_load_n(&t->stop, __ATOMIC_ACQUIRE)) {
    tail_flush(t);
    /* a heartbeat every ten seconds: its absence says the host is not being scheduled while
       the domain holds the core, which is a different finding from "the domain wrote nothing" */
    if (++ticks % 20 == 0)
      mark_u("SQ: tail alive, payload bytes=", t->metadata->length);
    usleep(500000);
  }
  return NULL;
}

/* Is `needle` anywhere in the first n bytes of the payload? Written out rather than calling memmem,
   which needs _GNU_SOURCE -- and a feature-test macro that silently is not set here would make this
   fail to compile, or worse, resolve to something else. The payload is not NUL-terminated. */
static int payload_has_marker(const char *payload, unsigned long n, const char *needle) {
  unsigned long m = (unsigned long)strlen(needle);
  unsigned long i;
  if (n == 0 || n > SQLITE_HC_REGION_SIZE || m == 0 || m > n)
    return 0;
  for (i = 0; i + m <= n; i++)
    if (memcmp(payload + i, needle, (size_t)m) == 0)
      return 1;
  return 0;
}

int main(int argc, char **argv) {
  int feature_probe = 0;   /* --feature-probe: ask the domain which restored APIs it carries */
  /* --slt IS AN EXPLICIT FLAG, NOT A THIRD POSITIONAL ARGUMENT. The optional argv[2] is
     strtoul'd as a probe stage, so a bare path would parse to 0 and quietly publish the
     stage-0 selector -- a run that looks like a staged probe and tests nothing. */
  const char *slt_path = 0;
  /* --speedtest1 "<args>": the whole speedtest1 command line as ONE argument, delivered to the
     domain in the payload's top half exactly as an SLT file is. A domain has no command line and
     speedtest1's option loop is the only way to set --testset and --size. */
  const char *speed_args = 0;
  unsigned long clamp_n = 0;
  unsigned long arena_bytes = 0;  /* --arena: region 2, the port's pool, linear */
  unsigned long pool_bytes = 0;   /* --pool: region 2, memsys5's heap, non-linear */
  unsigned long tables_bytes = 0; /* --tables: region 3, the tables beside the pool */
  region_id_t pool_region = (region_id_t)-1, tables_region = (region_id_t)-1;
  if (argc == 6 && !strcmp(argv[2], "--slt") && !strcmp(argv[4], "--clamp")) {
    slt_path = argv[3];
    clamp_n = strtoul(argv[5], NULL, 0);
  } else if (argc == 3 && !strcmp(argv[2], "--feature-probe")) {
    feature_probe = 1;
  } else if (argc == 4 && !strcmp(argv[2], "--slt")) {
    slt_path = argv[3];
  } else if (argc == 3 && !strcmp(argv[2], "--tail")) {
    tail_payload = 1;
  } else if (argc >= 5 && argc % 2 == 1 && !strcmp(argv[2], "--tail")) {
    /* --tail with the allocators' regions: --pool <bytes> (memsys5's heap, non-linear) or
       --arena <bytes> (the port's grant, linear), and --tables <bytes> for what sits beside */
    int i;
    tail_payload = 1;
    for (i = 3; i + 1 < argc; i += 2) {
      if (!strcmp(argv[i], "--arena"))
        arena_bytes = strtoul(argv[i + 1], NULL, 0);
      else if (!strcmp(argv[i], "--pool"))
        pool_bytes = strtoul(argv[i + 1], NULL, 0);
      else if (!strcmp(argv[i], "--tables"))
        tables_bytes = strtoul(argv[i + 1], NULL, 0);
      else {
        fprintf(stderr, "unknown option %s\n", argv[i]);
        return 2;
      }
    }
    if (arena_bytes && pool_bytes) {
      fprintf(stderr, "--arena and --pool exclude each other\n");
      return 2;
    }
    if (tables_bytes && !arena_bytes && !pool_bytes) {
      fprintf(stderr, "--tables needs --pool or --arena\n");
      return 2;
    }
  } else if (argc >= 4 && !strcmp(argv[2], "--speedtest1")) {
    /* EVERYTHING AFTER --speedtest1 IS THE BENCHMARK'S COMMAND LINE, joined with single spaces.
       One quoted string works too, and is what the QEMU script passes; accepting the split form as
       well is what makes the BOARD path safe. The board driver builds its invocation as a shell
       string ("{host} {host_args}") that then travels through the console, so a form whose meaning
       depends on quotes surviving three layers is a fragility with no upside. */
    static char joined[512];
    unsigned long at = 0;
    int i;
    /* THE REGION OPTIONS ARE LIFTED OUT BEFORE THE JOIN, and they have to be, because everything
       after --speedtest1 becomes the BENCHMARK's command line. Written naively, `--arena 4194304`
       here does not configure a region at all -- it is handed to speedtest1 as a speedtest1 flag,
       which either errors inside the domain or is ignored, and in both cases the arena silently is
       not the one asked for. The region creation and sharing further down is already common-path
       and keys on `arena_bytes || pool_bytes`, so nothing else needs teaching: this is option
       parsing only.

       Sublet needs --arena specifically (REV_BORROWED, the linear borrow under a handle the
       monitor keeps). SPEEDTEST1_REGION_ARENA is NOT a substitute -- it shares REV_SHARED, which
       the monitor delinearises, and a non-revocable share cannot carry the discipline. The two
       also claim the same slot and the refusal above enforces it. */
    for (i = 3; i + 1 < argc; ) {
      unsigned long *slot = !strcmp(argv[i], "--arena")  ? &arena_bytes
                          : !strcmp(argv[i], "--pool")   ? &pool_bytes
                          : !strcmp(argv[i], "--tables") ? &tables_bytes
                          : (unsigned long *)0;
      if (!slot) { i++; continue; }
      *slot = strtoul(argv[i + 1], NULL, 0);
      /* remove the pair from argv so the join below cannot see it */
      memmove(&argv[i], &argv[i + 2], (size_t)(argc - i - 2) * sizeof *argv);
      argc -= 2;
    }
    if (arena_bytes && pool_bytes) {
      fprintf(stderr, "%s: --arena and --pool exclude each other\n", argv[0]);
      return 2;
    }
    if (tables_bytes && !arena_bytes && !pool_bytes) {
      fprintf(stderr, "%s: --tables needs --pool or --arena\n", argv[0]);
      return 2;
    }
    for (i = 3; i < argc; i++) {
      unsigned long len = (unsigned long)strlen(argv[i]);
      if (at && at + 1 < sizeof joined)
        joined[at++] = ' ';
      if (at + len >= sizeof joined) {
        fprintf(stderr, "%s: --speedtest1 arguments exceed %lu bytes\n",
                argv[0], (unsigned long)sizeof joined);
        return 2;
      }
      memcpy(joined + at, argv[i], (size_t)len);
      at += len;
    }
    joined[at] = 0;
    speed_args = joined;
  } else if (argc != 2 && argc != 3) {
    fprintf(stderr, "usage: %s <sqlite-domain.dom> [probe-stage|--tail [--pool|--arena <bytes>] [--tables <bytes>]]\n"
                    "       %s <sqlite-domain.dom> --slt <file.test> [--clamp N]\n"
                    "       %s <sqlite-domain.dom> --feature-probe\n"
                    "       %s <sqlite-domain.dom> --speedtest1 --testset main --size 1\n",
                    argv[0], argv[0], argv[0], argv[0]);
    return 2;
  }
  /* REFUSE AN INVOCATION WITHOUT --testset, HERE, BEFORE A BOOT IS SPENT. The default testset is
     mix1, which contains json (omitted), rtree (not enabled), cte and star (decimal literals the
     tokenizer rejects under SQLITE_OMIT_FLOATING_POINT) and fp (needs round()). Any of those
     reaches fatal_error, so a default invocation is guaranteed to abort. That is a precondition,
     not a preference, and finding it out from a board transcript costs a stage. */
  if (speed_args && !strstr(speed_args, "--testset")) {
    fprintf(stderr, "%s: --speedtest1 args must name --testset explicitly; the default (mix1) "
                    "cannot run under this build's defines\n", argv[0]);
    return 2;
  }
  /* RUNTIME PROBE SELECTION (optional 2nd argument).
     Every probe used to be its OWN binary (-DCAPSTONE_SQLITE_STAGE=N), so each measurement
     drew a fresh ticket in the SHA5 stall lottery with a different image -- x101 lost that
     lottery 5 times running and its question is still unmeasured. Passing the stage at run
     time instead lets ONE domain image answer several questions, and lets the probe run in
     the image that is already known to enter. Omitted -> the domain keeps its built-in
     stage, so every existing invocation behaves exactly as before. */
  unsigned long probe_stage = 0;
  int have_probe_stage = 0;
  if (argc == 3 && !tail_payload && !feature_probe) {
    /* A POSITIVE TEST FOR A NUMBER, not a growing list of flags to exclude. argv[2] here is
       whatever the dispatch chain above did not consume, and strtoul() answers 0 for every
       non-numeric string -- so any flag-shaped argument reaching this point silently becomes
       "probe stage 0", and the run reports a stage nobody asked for. Nothing in the compiler
       or in a test of the flag's own feature can see it.
       --tail was rescued by a !tail_payload guard; --speedtest1 in its bare argc==3 form was
       not, and the board driver ships its invocation as a shell string through three quoting
       layers, which is exactly how `--speedtest1 ''` arrives as a bare `--speedtest1`.
       Requiring a digit ends the class instead of adding a third exclusion. */
    char *end = NULL;
    probe_stage = strtoul(argv[2], &end, 0);
    if (end == argv[2] || *end) {
      fprintf(stderr, "%s: unrecognised option %s\n", argv[0], argv[2]);
      return 2;
    }
    have_probe_stage = 1;
  }
#ifdef SPEEDTEST1_REGION_ARENA
  /* This build claims shared-region slot 2 for SQLite's arena at COMPILE time, so the runtime
     options that claim the same slot cannot also be honoured. Refuse here -- before capstone_init,
     before any region or domain exists, so there is nothing to unwind -- rather than ignoring them
     silently, which would hand the domain a capability set that looks right and is off by one. */
  if (arena_bytes || pool_bytes || tables_bytes) {
    fprintf(stderr, "%s: this host was built with -DSPEEDTEST1_REGION_ARENA, which already claims"
                    " shared-region slot 2; --arena/--pool/--tables claim the same slot\n", argv[0]);
    return 2;
  }
#endif

  if (capstone_init()) {
    mark("SQ: no-init\n");
    return 1;
  }

  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0)
    return fail_cleanup("create_dom failed", (unsigned long)domain);

  /* PHASE MARKERS. On silicon a monitor fault is C_PRINT + while(1), and C_PRINT
     goes to the RTL trace, not the UART -- so a wedge in create_dom and a wedge in
     the domain's entry glue look identical from the console: silence after
     libcapstone's last line. These two lines separate them, which is the whole
     difference between debugging the monitor and debugging the glue. */
  mark("SQ: A/dom-ok\n");
  mark_u("SQ: id=", (unsigned long)domain);

  /* FINE-GRAINED PHASE MARKERS. The board wedges somewhere between "create_dom ok"
     and "entering domain", and the monitor's region path has two silent `while(1)`s
     (split_out_cap: sbi_capstone.c:236, and :246 whose comment is "matching region.
     We don't support this for now"). SQLite is the first domain to create and share
     TWO regions, so no ladder rung covers this. One marker per call turns a 6-call
     gap into a single named culprit, which is worth one board run. */
  /* Attribute the faulting pc to an object. The first diagnosed run trapped ILLEGAL
     INSTRUCTION at (32-bit-truncated) 0xBDB30E3C during create_region -- neither the
     monitor (0x8001xxxx) nor the domain (0x10000). This host is a dynamic PIE, so print a
     known libc address and a known host-image address; subtracting identifies which
     mapping contains the fault, without parsing /proc/self/maps in a wedging process. */
#ifdef CAPSTONE_TRACE_ARM
  /* ARM THE COMMIT TRACER FROM THE RUNNING CORE, not from GDB.
   *
   * The tracer (core/tracer.sv, group 2 = every LDC/STC commit with its PC and its real
   * tag bit) is already in the flashed bitstream, and RTL simulation proves capture fires
   * when the group mask is set: four capability accesses, four captures, varying tag bits.
   * On the board it captured NOTHING across three boots -- and the one difference was the
   * arming route. There the mask went in over GDB, and the readback was taken at the same
   * halt as the write, which cannot tell the hardware register from the debugger's copy.
   *
   * Here the write is an ordinary csrw executed by this process, exactly as in the sim
   * that works. CSR 0x810 has bits[9:8] == 00, i.e. U-mode accessible, and CVA6 enforces
   * precisely that (privilege_violation tests access_priv < csr_addr priv_lvl, never true
   * for priv_lvl 0), so userspace may write it.
   *
   * This is in the HOST, deliberately, not in the domain. Every probe ever added INSIDE a
   * domain for this bug has made the fault disappear -- probed builds complete ~4/4 while
   * the un-probed build wedges 5/5 -- so an in-domain arm would buy observability by
   * destroying the thing being observed. The host runs in Linux userspace before capenter,
   * touches no domain image, and trace_enable_q is cleared only by hardware reset, so
   * arming here survives into the domain.
   *
   * The readback is printed rather than assumed: it is a read of the real CSR by the real
   * core, and it is the measurement that was missing on all three board boots. */
  {
    unsigned long _tm = (unsigned long)(CAPSTONE_TRACE_ARM), _tb = 0;
    __asm__ volatile("csrw 0x810, %0" :: "r"(_tm));
    __asm__ volatile("csrr %0, 0x810" : "=r"(_tb));
    mark_u("SQ: tracearm=", _tb);
  }
#endif
#ifdef CAPSTONE_TRACE_WP
  /* ARM THE STORE WATCHPOINT (CSR 0x811) at the subject stack slot.
   *
   * Group 9 logs the 64-bit value a committed store WROTE to the watched address. That is
   * the half `tval` cannot give: tval says the reload RETURNED zero, this says what the
   * spill PUT there. Non-zero here with zero there means the value was lost in between.
   *
   * Selective by ADDRESS rather than by opcode class, which is why it works where group 2
   * cannot: the monitor's trap-entry LDC fires on every timer tick and scavenges a
   * 256-entry ring within the interval between the domain stopping and the dump -- measured,
   * and identical on a wedging and a non-wedging arm. Nothing the monitor does touches this
   * address, so the ring holds only subject stores.
   *
   * THE ADDRESS MUST BE A GRANULE BASE. The comparator is word-granular
   * (st_commit_paddr[PLEN-1:3]) and a capability store presents ONE queue entry carrying the
   * granule base, so an address in the granule's upper half compares word 1 against word 0
   * and silently never fires -- returning empty, which would read as "no store happened".
   * The slot is s0-0x70 and capability stores are 16-byte aligned, so it is a granule base;
   * the assert below is here because that is a load-bearing coincidence, not a guarantee.
   *
   * Compiled in rather than passed as an argument: this value is derived from a previous
   * boot's wedge and is specific to one experiment, so binding it to the binary keeps the
   * two from drifting apart silently. 0x811 is U-mode accessible (bits[9:8] == 00) exactly
   * as 0x810 is, so no monitor or debugger is involved. */
  {
    unsigned long _wp = (unsigned long)(CAPSTONE_TRACE_WP), _wb = 0;
    if (_wp & 0xFUL)
      mark("SQ: tracewp=MISALIGNED -- not a granule base, the watchpoint cannot match\n");
    __asm__ volatile("csrw 0x811, %0" :: "r"(_wp));
    __asm__ volatile("csrr %0, 0x811" : "=r"(_wb));
    mark_u("SQ: tracewp=", _wb);
  }
#endif
  mark_u("SQ: libc=", (unsigned long)(void *)&printf);
  mark_u("SQ: self=", (unsigned long)(void *)&main);
  mark("SQ: B/mkregion1\n");
  region_id_t metadata_region = create_region(SQLITE_HC_REGION_SIZE);
  mark("SQ: C/mkregion2\n");
  region_id_t payload_region = create_region(SQLITE_HC_REGION_SIZE);
#ifdef SPEEDTEST1_REGION_ARENA
  /* THE THIRD REGION IS SQLite's ARENA, and it is deliberately NEVER MAPPED here. The host has no
     business reading it, and sqlite_host_row3_b2.c records that mapping an arena it does not touch
     is what made an earlier probe's failure ambiguous. Creating it is enough: the domain reaches it
     through the grant, and reads its size off the grant's own bounds. */
  mark("SQ: C2/mkarena\n");
  region_id_t arena_region = create_region(SPEEDTEST1_ARENA_SIZE);
  if ((long)arena_region < 0)
    return fail_cleanup("create_region(arena) failed -- above 4 MiB this needs a CMA area",
                        (unsigned long)SPEEDTEST1_ARENA_SIZE);
  mark_u("SQ: arena_bytes=", (unsigned long)SPEEDTEST1_ARENA_SIZE);
#endif
  mark("SQ: D/mapped\n");
  mark_u("SQ: r1=", (unsigned long)metadata_region);
  mark_u("SQ: r2=", (unsigned long)payload_region);
  /* TEST THE CREATE, not just the map. Without this a failed create returns
     (region_id_t)-1, map_region walks ids upward and returns NULL, and the run reports
     "map_region failed" -- which is what mislabelled the 64 MiB arm in three documents.
     The real failure was the buddy allocator's order-10 (4 MiB) wall at CREATE time.
     sqlite_host_row3_b2.c:57 already checks its arena this way; these two did not. */
  if ((long)metadata_region < 0 || (long)payload_region < 0)
    return fail_cleanup("create_region failed -- above 4 MiB this needs a CMA area",
                        (unsigned long)SQLITE_HC_REGION_SIZE);
  struct sqlite_hostcall_v0 *metadata =
      (struct sqlite_hostcall_v0 *)map_region(metadata_region,
                                              SQLITE_HC_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, SQLITE_HC_REGION_SIZE);
  /* map_region returns mmap()'s value raw, so a rejected mapping is MAP_FAILED
     ((void*)-1), NOT NULL. A bare !metadata test silently accepts it and the failure
     surfaces later as a fault on first use. */
  if (!metadata || metadata == (void *)-1 || !payload || payload == (void *)-1)
    return fail_cleanup("map_region failed", 0);

  memset(metadata, 0, SQLITE_HC_REGION_SIZE);
  memset(payload, 0, SQLITE_HC_REGION_SIZE);
  /* THE REGION SIZE THE HOST ACTUALLY USED, published for the domain to check against its
     own. Both halves take it from one #define, but they are separate compilations with
     separate -D flags, and a drift between them is silent and destructive: the host maps
     N bytes while the domain bounds its writes by M. Written unconditionally so the gate
     covers every build, not only SLT ones; nothing else reads this field. */
  metadata->result = (sqlite_hostcall_s64_t)SQLITE_HC_REGION_SIZE;
  /* VDBE clamp, published in `phase` -- unused by anything else. Lets one image bisect many
     clamp values instead of one firmware rebuild per value. */
  metadata->phase = (sqlite_hostcall_u64_t)clamp_n;
  /* Publish the probe selector AFTER the memset and BEFORE the domain runs. Magic-guarded so
     an unset region is indistinguishable from today's behaviour. */
  if (have_probe_stage)
    metadata->opcode = 0x5A6E0000UL | (probe_stage & 0xffUL);

  if (slt_path) {
    /* Load one SQLLogicTest file into the TOP HALF of the payload region. The domain
       parses it in place; the bottom half is where its report comes back.
       READ INTO A HEAP BUFFER FIRST, THEN COPY. Reading straight into the mapped region
       failed on the first read(2) with got=0 -- the kernel declines to write into that
       mapping, which is a property of the shared-region mapping and not of the file. A
       plain userspace memcpy into it works, and the file is bounded by the region's input
       half anyway, so the extra buffer costs nothing that matters. */
    unsigned long got = 0;
    char *staging;
    int fd = open(slt_path, O_RDONLY);
    if (fd < 0)
      return fail_cleanup("slt open failed", (unsigned long)errno);
    staging = (char *)malloc((size_t)SQLITE_HC_SLT_MAX_INPUT);
    if (!staging) { close(fd); return fail_cleanup("slt malloc failed", 0); }
    for (;;) {
      ssize_t n = read(fd, staging + got, (size_t)(SQLITE_HC_SLT_MAX_INPUT - got));
      if (n < 0) { close(fd); free(staging);
                   return fail_cleanup("slt read failed", (unsigned long)errno); }
      if (n == 0) break;
      got += (unsigned long)n;
      /* A FILE THAT DOES NOT FIT IS AN ERROR, NEVER A TRUNCATED ONE. Truncation would
         cut the input mid-record and the domain would report a smaller record count
         that still looks like a clean pass. */
      if (got >= SQLITE_HC_SLT_MAX_INPUT) {
        char probe;
        ssize_t more = read(fd, &probe, 1);
        if (more > 0) { close(fd); free(staging);
                        return fail_cleanup("slt file exceeds the region", got); }
        break;
      }
    }
    close(fd);
    if (got == 0) { free(staging); return fail_cleanup("slt file is empty", 0); }
    memcpy(payload + SQLITE_HC_SLT_INPUT_OFF, staging, (size_t)got);
    free(staging);
    metadata->opcode = SQLITE_HC_OP_SLT;
    metadata->offset = (sqlite_hostcall_u64_t)got;
    mark_u("SQ: slt=", got);
  }
  if (feature_probe) {
    metadata->opcode = SQLITE_HC_OP_FEATURE;
    mark("SQ: feature-probe\n");
  }
  if (speed_args) {
    /* Same top-half layout as an SLT file, and the same reason for the bound: the domain's output
       grows from offset 0 and must not meet the input. Refuse an oversized argument string rather
       than truncate it -- a truncated command line silently changes what was measured. */
    unsigned long n = (unsigned long)strlen(speed_args);
    if (n == 0)
      return fail_cleanup("speedtest1 args are empty", 0);
    if (n >= SQLITE_HC_SPEED_MAX_ARGS)
      return fail_cleanup("speedtest1 args exceed the region's input half", n);
    memcpy(payload + SQLITE_HC_SPEED_ARGS_OFF, speed_args, (size_t)n);
    metadata->opcode = SQLITE_HC_OP_SPEED;
    metadata->offset = (sqlite_hostcall_u64_t)n;
    mark_u("SQ: speedtest1=", n);
  }
  mark("SQ: E/share1\n");
  shared_region_annotated(domain, metadata_region,
                          SQLITE_HC_ANNOTATION_PERM_INOUT,
                          SQLITE_HC_ANNOTATION_REV_SHARED);
  mark("SQ: F/share2\n");
  shared_region_annotated(domain, payload_region,
                          SQLITE_HC_ANNOTATION_PERM_INOUT,
                          SQLITE_HC_ANNOTATION_REV_SHARED);
  /* SLOT 2 IS CLAIMED BY EXACTLY ONE OF THESE TWO, AND THEY ARE NOT ADDITIVE -- which is why this
     is an #else and not a second block. Both domains key their captures on the ORDER the host
     shares in: 0 metadata, 1 payload, 2 the allocator's memory. speedtest1_measure.c takes slot 2
     as the arena, speedtest1_domain.c takes it as the pool and slot 3 as the tables.

     What running both sequences would actually do, stated precisely because the first version of
     this comment overstated it symmetrically: the arena share below is textually first, so a
     both-sequences build shares arena=2, pool=3, tables=4. speedtest1_measure.c still finds its
     arena at 2 and ignores 3 and 4 -- a `--pool N` the caller asked for is silently IGNORED, the
     region created and wasted. speedtest1_domain.c is the one handed the wrong capability, taking
     the arena as its pool. Neither outcome is visible to the compiler, and a silently ignored
     --pool is reason enough for the refusal further up. */
#ifdef SPEEDTEST1_REGION_ARENA
  /* THIRD, because domain_main keys on the capture ORDER: 0 metadata, 1 payload, 2 arena. Sharing
     it anywhere else in this sequence silently hands SQLite's heap to the wrong pointer. */
  mark("SQ: F2/share3\n");
  shared_region_annotated(domain, arena_region,
                          SQLITE_HC_ANNOTATION_PERM_INOUT,
                          SQLITE_HC_ANNOTATION_REV_SHARED);
#else
  /* The allocators' memory, regions above the payload's, never touched from here: once the
     domain has revoked a lineage in a region, a host access to those pages aborts QEMU
     (sqlite_host_row3.c). Both are shared with a handle the monitor keeps (REV_BORROWED for
     the linear pool of the port, REV_DEFAULT for the non-linear ones), so that after the run
     release_region() can revoke them, pop them and give the memory back. Above 4 MiB a region
     comes from the kernel's CMA area (modcapstone create_region; cma= on the command line). */
  if (arena_bytes || pool_bytes) {
    mark("SQ: F2/mkregion3\n");
    pool_region = create_region(arena_bytes ? arena_bytes : pool_bytes);
    mark_u("SQ: r3=", (unsigned long)pool_region);
    if ((long)pool_region < 0)
      return fail_cleanup("create_region for the pool failed (a region above 4 MiB needs cma=)", arena_bytes ? arena_bytes : pool_bytes);
    shared_region_annotated(domain, pool_region, SQLITE_HC_ANNOTATION_PERM_INOUT,
                            arena_bytes ? SQLITE_HC_ANNOTATION_REV_BORROWED
                                        : SQLITE_HC_ANNOTATION_REV_DEFAULT);
    mark_u(arena_bytes ? "SQ: arena=" : "SQ: pool=", arena_bytes ? arena_bytes : pool_bytes);
  }
  if (tables_bytes) {
    mark("SQ: F3/mkregion4\n");
    tables_region = create_region(tables_bytes);
    mark_u("SQ: r4=", (unsigned long)tables_region);
    if ((long)tables_region < 0)
      return fail_cleanup("create_region for the tables failed", tables_bytes);
    shared_region_annotated(domain, tables_region, SQLITE_HC_ANNOTATION_PERM_INOUT,
                            SQLITE_HC_ANNOTATION_REV_DEFAULT);
    mark_u("SQ: tables=", tables_bytes);
  }
#endif

  mark("SQ: G/enter\n");
  /* --tail: a thread prints the payload WHILE the domain runs, from where it left off, every
     half second. A domain that never returns (a wedge, a fatal error parked in abort()) then
     still shows everything it wrote up to that point, and a long benchmark shows its
     progress. Both regions are shared with the domain, so the length it publishes and the
     bytes behind it are visible here as they land. */
  struct tail_state tail = {metadata, payload, 0, 0};
  pthread_t tail_thread;
  int tail_started = tail_payload && pthread_create(&tail_thread, NULL, tail_main, &tail) == 0;
  unsigned long result = call_dom(domain);
  if (tail_started) {
    __atomic_store_n(&tail.stop, 1, __ATOMIC_RELEASE);
    pthread_join(tail_thread, NULL);
  }
  mark("SQ: H/return\n");
  if (metadata->length > tail.printed && metadata->length <= SQLITE_HC_REGION_SIZE) {
    (void)write(STDOUT_FILENO, payload + tail.printed, (size_t)(metadata->length - tail.printed));
    fflush(stdout);
  }
  /* the allocators' regions go back, newest first: revoked in the monitor, popped, freed */
  if ((long)tables_region >= 0)
    mark_u("SQ: released tables rc=", (unsigned long)release_region(tables_region));
  if ((long)pool_region >= 0)
    mark_u("SQ: released pool rc=", (unsigned long)release_region(pool_region));
  /* AN SLT RUN MUST RETURN ITS OWN MARKER, AND NOTHING ELSE COUNTS -- including DONE.
     DONE here would mean the SLT dispatch never fired and the ordinary workload ran
     instead, which prints its own markers and would otherwise look like a success. The
     0x5117BADn values are the runner's refusals-to-start; each means no records were
     evaluated, and each must fail the run rather than pass it with an empty report. */
  if (feature_probe) {
    /* The probe returns the 0x4EB0 marker family the driver already reads as a result, so no
       driver change is needed; the FEATURE-SET line itself comes back in the payload. */
    if ((result & 0xFFF00000UL) != 0x4EB00000UL)
      return fail_cleanup("feature probe did not run", result);
    mark_u("SQ: feature=", result);
  } else if (slt_path) {
    /* A DOMAIN-SIDE PROBE MARKER IS A SUCCESSFUL RUN, NOT A FAILURE TO START.
     *
     * Probe builds (CAPSTONE_HEAPCAP_PROBE and friends) return their own 0x4EA0/0x4EB0 marker
     * instead of SQLITE_HC_SLT_RAN, because they deliberately run INSTEAD of the workload.
     * Treating that as a failure made the host exit 1, and the board driver's "no RESULT
     * retval= marker plus a non-zero exit" heuristic then declared HARD STOP -- "the domain
     * almost certainly was not staged" -- and abandoned the remaining arms of the boot. It was
     * staged and it ran: the monitor echoed the value back as ENT2:4EA00A01. That cost two
     * arms of a control-validated board session, which is expensive for a classification
     * error. Print the marker and exit cleanly so the driver reads the arm as what it is. */
    if ((result & 0xFFF00000UL) == 0x4EA00000UL ||
        (result & 0xFFF00000UL) == 0x4EB00000UL) {
      mark_u("SQ: probe=", result);
    } else if (result != SQLITE_HC_SLT_RAN) {
      return fail_cleanup("slt did not run", result);
    }
  } else if (speed_args) {
    /* THREE OUTCOMES, AND THE PAYLOAD IS WHAT SEPARATES TWO OF THEM.
     *
     * A clean run returns the 0x4EB1 marker. A run that hit speedtest1's fatal_error cannot return
     * at all: exit() is unreachable in a domain, so the abort path writes its report and then takes
     * a deliberate capability fault, and the monitor hands back CAPSTONE_DOMAIN_FAULT_RETVAL
     * (0x0FA017ED). A GENUINE capability defect returns exactly the same value -- the retval cannot
     * tell them apart, and neither can the monitor's tags. The marker the domain wrote into the
     * payload BEFORE faulting is the only discriminator, which is why the payload is dumped above
     * unconditionally rather than only on success.
     *
     * Both failure modes exit non-zero; they are distinguished by the message so a transcript reads
     * as what it is instead of as "the domain was never staged". */
    if ((result & 0xFFF00000UL) == 0x4EB00000UL) {
      mark_u("SQ: speedtest1-ran=", result);
    } else if (result == 0x0FA017EDUL) {
      unsigned long n = (unsigned long)metadata->length;
      int reported = payload_has_marker(payload, n, "__CAPSTONE_SPEEDTEST1_ABORTED__");
      return fail_cleanup(reported ? "speedtest1 aborted (fatal_error; see the report above)"
                                   : "domain faulted with no speedtest1 report",
                          result);
    } else {
      return fail_cleanup("speedtest1 did not run", result);
    }
  } else if (result != SQLITE_HC_RET_DONE) {
    return fail_cleanup("unexpected domain return", result);
  }

  if (capstone_cleanup()) {
    mark("SQ: no-cleanup\n");
    return 1;
  }
  return 0;
}
