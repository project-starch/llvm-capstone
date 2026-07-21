/* Freestanding borrow-cost controller -- FPGA/CapliFive silicon variant,
 * GLOBAL-FREE / SINGLE-REGION protocol (2026-07-20).
 *
 * WHY A SEPARATE CONTROLLER: the sibling borrow_cost_fpga_ctl.c drives the old
 * two-region + CALL protocol (LINEAR arena REV_TRANSFERRED, results REV_SHARED,
 * measure during the CALL). That domain tracked which region is which across the
 * two SHARE entries and the CALL using module statics -- and our LLVM Capstone
 * backend reaches statics via `cincoffset X, gp` assuming gp = PCC(cursor 0), a
 * form the RTL does not fabricate (only our QEMU fork does). On silicon that
 * domain faults (gp=0 -> `delin gp` stalls). See history 20-07-2026 + NDSS plan.
 *
 * This controller pairs with the GLOBAL-FREE domain borrow_cost_fpga_nogp.dom
 * (start-fpga-nogp.S + borrow_cost_fpga_nogp.c), which has NO cross-entry state:
 * the whole measurement runs inside ONE REGION_SHARE entry, on a single
 * host-retained region that is BOTH scratch and results. So the protocol here is
 * a single share, no CALL:
 *   1. create ONE region (REV_SHARED, host retains its mapping);
 *   2. share it once (PERM_INOUT, REV_SHARED) -- this is the domain entry that
 *      runs the measurement and writes the eight result slots into the region;
 *   3. read the eight results straight back out of the same mapping;
 *   4. print the per-op cycle costs as RESULT lines for the UART parser.
 *
 * FP/glibc note carries over: like the sibling this links NO glibc (the board
 * rejects glibc's hard-float `fsd`), so it is built -nostdlib -static
 * -march=rv64imac -mabi=lp64 (soft-float ABI, zero FP), own _start, raw Linux
 * syscalls via ecall, integer-only output, same ioctl protocol as libcapstone.
 */

#include "../runtime-qemu/borrow-cost-probe/borrow_cost_probe.h"

/* ------------------------------------------------------------------ *
 *  Raw Linux syscalls (RISC-V 64, generic ABI) -- no libc.
 * ------------------------------------------------------------------ */
typedef unsigned long ulong;
typedef long ssize_t_;

static inline long _sys(long n, long a0, long a1, long a2,
                        long a3, long a4, long a5) {
  register long ra0 __asm__("a0") = a0;
  register long ra1 __asm__("a1") = a1;
  register long ra2 __asm__("a2") = a2;
  register long ra3 __asm__("a3") = a3;
  register long ra4 __asm__("a4") = a4;
  register long ra5 __asm__("a5") = a5;
  register long rn  __asm__("a7") = n;
  __asm__ volatile("ecall"
                   : "+r"(ra0)
                   : "r"(ra1), "r"(ra2), "r"(ra3), "r"(ra4), "r"(ra5), "r"(rn)
                   : "memory");
  return ra0;
}
#define SYS_openat      56
#define SYS_close       57
#define SYS_lseek       62
#define SYS_read        63
#define SYS_write       64
#define SYS_ioctl       29
#define SYS_mmap       222
#define SYS_munmap     215
#define SYS_exit_group  94

#define AT_FDCWD        (-100)
#define O_RDONLY        0
#define O_RDWR          2
#define O_NONBLOCK      04000
#define SEEK_END        2
#define PROT_READ       1
#define PROT_WRITE      2
#define MAP_SHARED      1
#define MAP_PRIVATE     2
#define MAP_ANONYMOUS   0x20
#define MAP_FAILED      ((void *)-1)

static int sys_open(const char *p, int flags) {
  return (int)_sys(SYS_openat, AT_FDCWD, (long)p, flags, 0, 0, 0);
}
static int sys_close(int fd) { return (int)_sys(SYS_close, fd, 0, 0, 0, 0, 0); }
static long sys_lseek(int fd, long off, int wh) {
  return _sys(SYS_lseek, fd, off, wh, 0, 0, 0);
}
static long sys_write(int fd, const void *b, ulong n) {
  return _sys(SYS_write, fd, (long)b, (long)n, 0, 0, 0);
}
static long sys_ioctl(int fd, ulong req, void *arg) {
  return _sys(SYS_ioctl, fd, (long)req, (long)arg, 0, 0, 0);
}
static void *sys_mmap(void *addr, ulong len, int prot, int flags, int fd, long off) {
  return (void *)_sys(SYS_mmap, (long)addr, (long)len, prot, flags, fd, off);
}
static long sys_munmap(void *addr, ulong len) {
  return _sys(SYS_munmap, (long)addr, (long)len, 0, 0, 0, 0);
}
static void sys_exit(int code) { _sys(SYS_exit_group, code, 0, 0, 0, 0, 0); }

static int mmap_ok(void *p) { return (ulong)p <= (ulong)-4096UL; }

/* ------------------------------------------------------------------ *
 *  Tiny libc.
 * ------------------------------------------------------------------ */
static void *memset_(void *d, int c, ulong n) {
  unsigned char *p = d;
  while (n--) *p++ = (unsigned char)c;
  return d;
}
static void *memcpy_(void *d, const void *s, ulong n) {
  unsigned char *dp = d;
  const unsigned char *sp = s;
  while (n--) *dp++ = *sp++;
  return d;
}
static int memeq_(const void *a, const void *b, ulong n) {
  const unsigned char *x = a, *y = b;
  while (n--) if (*x++ != *y++) return 0;
  return 1;
}

/* ------------------------------------------------------------------ *
 *  Integer-only output (no printf, no FP).
 * ------------------------------------------------------------------ */
static void puts_(const char *s) {
  ulong n = 0;
  while (s[n]) n++;
  sys_write(1, s, n);
}
static void putu_(ulong v) {         /* decimal */
  char buf[24];
  int i = 24;
  if (v == 0) buf[--i] = '0';
  while (v) { buf[--i] = (char)('0' + v % 10); v /= 10; }
  sys_write(1, buf + i, (ulong)(24 - i));
}
/* ------------------------------------------------------------------ *
 *  ioctl protocol -- copied verbatim from capstone.h.
 * ------------------------------------------------------------------ */
#define IOC_MAGIC 0xb8u
#define _IOC(dir, type, nr, size) \
  (((ulong)(dir) << 30) | ((ulong)(size) << 16) | ((ulong)(type) << 8) | (ulong)(nr))
#define _IOC_WR 3u
#define _IOWR(type, nr, sz) _IOC(_IOC_WR, (type), (nr), (sz))

typedef ulong dom_id_t;
typedef ulong region_id_t;
typedef ulong size_t_;

struct ioctl_dom_create_args {
  void *code_begin; size_t_ code_len; size_t_ entry_offset;
  void *s_load_begin; size_t_ s_load_len; size_t_ s_entry_offset; size_t_ s_size;
  dom_id_t dom_id;
};
struct ioctl_dom_call_args { dom_id_t dom_id; ulong retval; };
struct ioctl_region_create_args { size_t_ len; region_id_t region_id; size_t_ mmap_offset; };
struct ioctl_region_share_annotated_args {
  dom_id_t dom_id; region_id_t region_id;
  ulong annotation_perm; ulong annotation_rev; unsigned retval;
};
struct ioctl_region_query_args { region_id_t region_id; size_t_ mmap_offset; size_t_ len; };

#define IOCTL_DOM_CREATE  _IOWR(IOC_MAGIC, 0, sizeof(struct ioctl_dom_create_args))
#define IOCTL_DOM_CALL    _IOWR(IOC_MAGIC, 1, sizeof(struct ioctl_dom_call_args))
#define IOCTL_REGION_CREATE _IOWR(IOC_MAGIC, 2, sizeof(struct ioctl_region_create_args))
#define IOCTL_REGION_QUERY  _IOWR(IOC_MAGIC, 4, sizeof(struct ioctl_region_query_args))
#define IOCTL_REGION_SHARE_ANNOTATED \
  _IOWR(IOC_MAGIC, 7, sizeof(struct ioctl_region_share_annotated_args))

/* ------------------------------------------------------------------ *
 *  Minimal ELF64 (only the fields the loader touches).
 * ------------------------------------------------------------------ */
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long  u64;
struct Ehdr {
  unsigned char e_ident[16]; u16 e_type; u16 e_machine; u32 e_version;
  u64 e_entry; u64 e_phoff; u64 e_shoff; u32 e_flags;
  u16 e_ehsize; u16 e_phentsize; u16 e_phnum;
  u16 e_shentsize; u16 e_shnum; u16 e_shstrndx;
};
struct Phdr {
  u32 p_type; u32 p_flags; u64 p_offset; u64 p_vaddr; u64 p_paddr;
  u64 p_filesz; u64 p_memsz; u64 p_align;
};
#define PT_LOAD 1
#define PF_X    1
#define EM_RISCV 243
#define EM_CAPSTONE 259
static const unsigned char ELF_MAGIC[4] = {0x7f, 'E', 'L', 'F'};

/* ------------------------------------------------------------------ *
 *  Domain / region ops (libcapstone port, no QEMU debug-counter insn).
 * ------------------------------------------------------------------ */
#define TAG "borrow-breakdown-fpga"
#define MAX_REGION_N 64
#define MAP_SIZE_LIMIT 0x10000000UL

static int dev_fd;
static size_t_ region_mmap_offsets[MAX_REGION_N];
static int region_mmappable[MAX_REGION_N];
static int region_n;

struct ElfCode { void *map_base; ulong map_len; ulong code_start, code_len, entry_offset, loadable_size; };

static int load_elf_code(const char *path, struct ElfCode *res) {
  int fd = sys_open(path, O_RDONLY);
  if (fd < 0) { puts_(TAG ": open .dom failed\n"); return 1; }
  long fsize = sys_lseek(fd, 0, SEEK_END);
  if (fsize <= 0) { puts_(TAG ": empty .dom\n"); sys_close(fd); return 1; }
  struct Ehdr *eh = sys_mmap(0, (ulong)fsize, PROT_READ, MAP_SHARED, fd, 0);
  if (!mmap_ok(eh)) { puts_(TAG ": mmap .dom failed\n"); sys_close(fd); return 1; }
  if (!memeq_(ELF_MAGIC, eh->e_ident, 4)) { puts_(TAG ": not ELF\n"); goto bad; }
  if (eh->e_machine != EM_RISCV && eh->e_machine != EM_CAPSTONE) { puts_(TAG ": not RV/Capstone\n"); goto bad; }

  struct Phdr *ph = (struct Phdr *)((char *)eh + eh->e_phoff);
  int phnum = eh->e_phnum, i;
  int exec_idx = -1, first_idx = -1;
  ulong lstart = 0, lend = 0;
  for (i = 0; i < phnum; i++) {
    if (ph[i].p_type != PT_LOAD) continue;
    if (first_idx == -1 || ph[i].p_vaddr < ph[first_idx].p_vaddr) { first_idx = i; lstart = ph[i].p_vaddr; }
    if (exec_idx == -1 && (ph[i].p_flags & PF_X)) exec_idx = i;
    ulong e = ph[i].p_vaddr + ph[i].p_memsz;
    if (e > lend) lend = e;
  }
  if (exec_idx == -1 || first_idx == -1 || lend <= lstart) { puts_(TAG ": no PT_LOAD/exec seg\n"); goto bad; }
  ulong entry = eh->e_entry;
  if (entry < ph[exec_idx].p_vaddr ||
      entry >= ph[exec_idx].p_vaddr + ph[exec_idx].p_filesz) { puts_(TAG ": entry OOB\n"); goto bad; }

  ulong image_size = lend - lstart;
  unsigned char *img = sys_mmap(0, image_size, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (!mmap_ok(img)) { puts_(TAG ": image mmap failed\n"); goto bad; }
  memset_(img, 0, image_size);
  for (i = 0; i < phnum; i++) {
    if (ph[i].p_type != PT_LOAD) continue;
    ulong off = ph[i].p_vaddr - lstart;
    if (off + ph[i].p_filesz > image_size || off + ph[i].p_memsz > image_size) {
      puts_(TAG ": seg OOB\n"); sys_munmap(img, image_size); goto bad;
    }
    memcpy_(img + off, (char *)eh + ph[i].p_offset, ph[i].p_filesz);
  }
  sys_munmap(eh, (ulong)fsize);
  sys_close(fd);
  res->map_base = img; res->map_len = image_size;
  res->code_start = (ulong)img; res->code_len = image_size;
  res->entry_offset = entry - lstart; res->loadable_size = image_size;
  return 0;
bad:
  sys_munmap(eh, (ulong)fsize);
  sys_close(fd);
  return 1;
}

static dom_id_t create_dom(const char *c_path) {
  struct ElfCode c;
  if (load_elf_code(c_path, &c)) return (dom_id_t)-1;
  struct ioctl_dom_create_args a;
  memset_(&a, 0, sizeof(a));
  a.code_begin = (void *)c.code_start;
  a.code_len = c.code_len;
  a.entry_offset = c.entry_offset;
  a.s_load_len = 0;
  a.dom_id = (dom_id_t)-1;
  long r = sys_ioctl(dev_fd, IOCTL_DOM_CREATE, &a);
  sys_munmap(c.map_base, c.map_len);
  if (r) return (dom_id_t)-1;
  return a.dom_id;
}

static region_id_t create_region(ulong len) {
  struct ioctl_region_create_args a;
  memset_(&a, 0, sizeof(a));
  a.len = len; a.region_id = (region_id_t)-1;
  sys_ioctl(dev_fd, IOCTL_REGION_CREATE, &a);
  return a.region_id;
}

static void *map_region(region_id_t id, ulong len) {
  while (region_n <= (int)id) {
    struct ioctl_region_query_args q;
    memset_(&q, 0, sizeof(q));
    q.region_id = region_n;
    sys_ioctl(dev_fd, IOCTL_REGION_QUERY, &q);
    if (q.len == 0) return 0;
    region_mmap_offsets[region_n] = q.mmap_offset;
    region_mmappable[region_n] = q.len < MAP_SIZE_LIMIT;
    region_n++;
  }
  if (!region_mmappable[id]) return 0;
  void *p = sys_mmap(0, len, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd,
                     (long)region_mmap_offsets[id]);
  return mmap_ok(p) ? p : 0;
}

static void shared_region_annotated(dom_id_t dom, region_id_t reg,
                                    ulong perm, ulong rev) {
  struct ioctl_region_share_annotated_args a;
  memset_(&a, 0, sizeof(a));
  a.dom_id = dom; a.region_id = reg; a.annotation_perm = perm; a.annotation_rev = rev;
  sys_ioctl(dev_fd, IOCTL_REGION_SHARE_ANNOTATED, &a);
}

/* results region: REV_SHARED (host retains its mapping). */
#define REV_SHARED 0x2u
#define CAPSTONE_DEV_PATH_STR "/dev/capstone"

static int run(const char *dom_path) {
  dev_fd = sys_open(CAPSTONE_DEV_PATH_STR, O_NONBLOCK | O_RDWR);
  if (dev_fd < 0) { puts_(TAG ": open /dev/capstone failed\n"); return 1; }

  dom_id_t dom = create_dom(dom_path);
  if ((long)dom < 0) { puts_(TAG ": create_dom failed\n"); return 1; }
  puts_(TAG ": created domain ID = "); putu_(dom); puts_("\n");

  /* Single REV_SHARED region: both scratch (raw_src/copy_dst, at fixed offsets
   * inside the region) AND the eight result slots at the base. Host retains its
   * mapping, so it reads the domain's writes straight back after the share. */
  region_id_t region_id = create_region(BORROW_COST_REGION_SIZE);
  unsigned char *results = map_region(region_id, BORROW_COST_REGION_SIZE);
  if (!results) { puts_(TAG ": map region failed\n"); return 1; }
  memset_(results, 0, BORROW_COST_REGION_SIZE);
  puts_(TAG ": region = "); putu_(region_id);
  puts_(" (REV_SHARED, scratch+results)\n");

  /* The share IS the measurement: __domcallsaves(dom, REGION_SHARE, region)
   * enters the domain once with the region cap; domain_main runs every measure
   * and writes the eight slots. No CALL, no cross-entry state, no globals. */
  shared_region_annotated(dom, region_id, BORROW_COST_ANNOTATION_PERM_INOUT,
                          REV_SHARED);
  puts_(TAG ": region shared (host retains); measurement ran in share entry\n");

  /* Breakdown result slots (see borrow_breakdown_fpga_nogp.c BD_SLOT_*):
   * 0=iters 1=empty 2=raw(load) 3=mrev_only(mint) 4=mrd(mint+delin+revoke)
   * 5=full(borrow) 6=magic(0xB2EA) 7=sink. */
  const ulong *r = (const ulong *)results;
  ulong iters = r[0];
  ulong empty = r[1];
  ulong raw = r[2];
  ulong mrev_only = r[3];
  ulong mrd = r[4];
  ulong full = r[5];
  ulong magic = r[6];
  if (iters == 0) { puts_(TAG ": iters=0\n"); return 1; }
  if (magic != 0xB2EAUL) { puts_(TAG ": bad magic (not a breakdown domain)\n"); return 1; }

  puts_(TAG ": RAW iters="); putu_(iters);
  puts_(" empty="); putu_(empty);
  puts_(" raw="); putu_(raw);
  puts_(" mrev_only="); putu_(mrev_only);
  puts_(" mrd="); putu_(mrd);
  puts_(" full="); putu_(full); puts_("\n");

  /* Per-op cycle costs (empty-subtracted). Directly-measured quantities:
   *   load          = raw
   *   mrev+delin+revoke = mrd           (the reclaim unit; mrev is ~1 instr per
   *                                       the QEMU decomposition, so mrd ~= the
   *                                       delin+revoke tree cost)
   *   borrow        = full              (mrev + delin + revoke + load)
   * cross-check: full - mrd  should ~= raw (the load).
   * (mrev-in-isolation is omitted: the only loop that isolates it accumulates
   *  un-revoked nodes and resets this silicon -- so mrev_only == empty here.) */
  ulong load_pp = (raw > empty) ? (raw - empty) / iters : 0;
  ulong mrd_pp = (mrd > empty) ? (mrd - empty) / iters : 0;
  ulong full_pp = (full > empty) ? (full - empty) / iters : 0;
  ulong xchk_pp = (full_pp > mrd_pp) ? (full_pp - mrd_pp) : 0;
  (void)mrev_only;

  puts_(TAG ": RESULT cycles/op  load="); putu_(load_pp);
  puts_("  mrev+delin+revoke="); putu_(mrd_pp);
  puts_("  borrow(full)="); putu_(full_pp); puts_("\n");

  puts_(TAG ": RESULT xcheck   full-mrd="); putu_(xchk_pp);
  puts_(" (should ~= load="); putu_(load_pp); puts_(")\n");

  puts_(TAG ": measurement complete\n");
  return 0;
}

/* ------------------------------------------------------------------ *
 *  Entry: parse argc/argv off the stack, no libc runtime.
 * ------------------------------------------------------------------ */
void _start_c(long *sp) {
  long argc = sp[0];
  char **argv = (char **)(sp + 1);
  if (argc < 2) {
    puts_("usage: borrow_breakdown_fpga_nogp_ctl <borrow_breakdown_fpga_nogp.dom>\n");
    sys_exit(2);
  }
  int rc = run(argv[1]);
  sys_exit(rc);
}

__asm__(
  ".section .text\n"
  ".globl _start\n"
  "_start:\n"
  /* Initialise gp (global pointer). The compiler addresses small globals
   * gp-relative; without this, gp holds garbage and every global store faults.
   * norelax so `lla gp,__global_pointer$` isn't itself relaxed to gp-relative. */
  "  .option push\n"
  "  .option norelax\n"
  "  lla gp, __global_pointer$\n"
  "  .option pop\n"
  "  mv a0, sp\n"        /* pass stack pointer (points at argc) */
  "  andi sp, sp, -16\n" /* 16-byte align */
  "  call _start_c\n"
  "  ebreak\n"           /* _start_c never returns */
);
