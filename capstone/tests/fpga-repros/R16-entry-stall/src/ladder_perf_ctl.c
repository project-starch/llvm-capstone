/* Freestanding silicon-ladder perf controller (FPGA/CapliFive).
 *
 * Generic board controller for the silicon-ladder perf domains
 * (tests/runtime-qemu/silicon-ladder/<rung>_fpga_app.c built via
 * build-ladder-fpga.sh). It creates the domain from the given .dom, creates ONE
 * REV_SHARED region, and shares it once -- that share IS the domain entry, which
 * runs the rung's workload and writes three unsigned-long slots at the region
 * base (see ladder_perf_domain.h):
 *   res[0] = retval  (checksum; the caller compares it to the native oracle)
 *   res[1] = cycles  (mcycle delta across the compute)
 *   res[2] = 0xD09E  (ran-marker)
 * The controller reads them straight back out of the retained mapping and prints
 *   RESULT <name> retval=<r0> cycles=<r1> ran=<r2>
 * for the UART harvester. The DIAGNOSTIC rung (gp_diag) additionally fills
 * res[3..11] with one RAW value per probe; when any is non-zero the controller
 * also prints
 *   DEBUG <name> dbg0=<r3> dbg1=<r4> ...
 * which localizes a silicon miscompute to a specific mechanism in ONE run
 * (a checksum alone cannot -- the FNV fold is not injective). No CALL, no cross-entry state (the region cap is
 * domain_main's argument; the rung reaches its own globals via gp[i]).
 *
 * Same freestanding soft-float model as borrow_cost_fpga_nogp_ctl.c: links NO
 * glibc (the board rejects glibc's hard-float fsd), built -nostdlib -static
 * -march=rv64imac -mabi=lp64, own _start, raw Linux syscalls, integer-only I/O,
 * same ioctl protocol as libcapstone.
 */

/* ------------------------------------------------------------------ *
 *  Raw Linux syscalls (RISC-V 64, generic ABI) -- no libc.
 * ------------------------------------------------------------------ */
typedef unsigned long ulong;

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

/* Freestanding string compare: no libc here. Used to find .capstone_gp_initdesc among
   the section names. */
static int streq_(const char *a, const char *b) {
  while (*a && *b && *a == *b) { a++; b++; }
  return *a == *b;
}
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
struct ioctl_region_create_args { size_t_ len; region_id_t region_id; size_t_ mmap_offset; };
struct ioctl_region_share_annotated_args {
  dom_id_t dom_id; region_id_t region_id;
  ulong annotation_perm; ulong annotation_rev; unsigned retval;
};
struct ioctl_region_query_args { region_id_t region_id; size_t_ mmap_offset; size_t_ len; };

#define IOCTL_DOM_CREATE  _IOWR(IOC_MAGIC, 0, sizeof(struct ioctl_dom_create_args))
#define IOCTL_REGION_CREATE _IOWR(IOC_MAGIC, 2, sizeof(struct ioctl_region_create_args))
#define IOCTL_REGION_QUERY  _IOWR(IOC_MAGIC, 4, sizeof(struct ioctl_region_query_args))
#define IOCTL_REGION_SHARE_ANNOTATED \
  _IOWR(IOC_MAGIC, 7, sizeof(struct ioctl_region_share_annotated_args))

/* PERM_INOUT / REV_SHARED: same annotation the borrow-cost controller uses (host
 * retains the mapping so it reads the domain's writes straight back). */
#define ANNOT_PERM_INOUT 0x1u
#define REV_SHARED       0x2u
#define REGION_SIZE      4096UL
#define CAPSTONE_DEV_PATH_STR "/dev/capstone"

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
struct Shdr {
  u32 sh_name; u32 sh_type; u64 sh_flags; u64 sh_addr; u64 sh_offset;
  u64 sh_size; u32 sh_link; u32 sh_info; u64 sh_addralign; u64 sh_entsize;
};
#define PT_LOAD 1
#define PF_X    1
#define EM_RISCV 243
#define EM_CAPSTONE 259
static const unsigned char ELF_MAGIC[4] = {0x7f, 'E', 'L', 'F'};

#define TAG "ladder-perf"

/* res[3..3+LADDER_DBG_SLOTS) are the diagnostic rung's raw per-probe values.
   4 KiB region => 512 slots, so this is far inside bounds. Perf rungs leave the
   slots zero and the DEBUG line is then suppressed entirely, so raising this
   costs them nothing.

   45 is chosen to reach res[47]: it covers gp_diag4's four 8-word readback
   groups + canary (33), AND both of gp_diag3's seeded data windows in full --
   W1 = res[32..40) at dbg29..dbg36, W2 = res[40..48) at dbg37..dbg44.

   Overlapping a diagnostic rung's DATA window is deliberate here, not an
   accident: the controller memsets the region before the share and only reads
   these slots AFTER the domain returns, so it observes what the domain's stores
   actually left in memory without perturbing the domain at all. That is how we
   established (at 33 slots, covering W1[0..3]) that the stores LAND and the
   gp-captable fault is therefore load-side. */
#define LADDER_DBG_SLOTS 45

static int dev_fd;

struct ElfCode { void *map_base; ulong map_len; ulong code_start, code_len, entry_offset;
                 ulong globals_off; };

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
  /* GLOBALS OFFSET, from the section headers. The monitor needs to know where the
     globals region starts inside the image so it can copy the initializer blob and
     split gp at the right boundary. It defaults to 0x1000, which is right only for a
     4 KiB code window; a domain with a large .text (SQLite: globals at 0x140000) needs
     the real value or the monitor copies .text as though it were globals and splits PCC
     4 KiB in.
     .capstone_gp_initdesc is placed FIRST in the globals region by link-gpfree.ld, so
     its address minus the image base IS the offset. Read from section headers rather
     than symbols: no symtab walk, and the section is KEEP'd so --gc-sections cannot
     drop it. Same mechanism libcapstone uses; the ladder controller lacked it, which is
     why no ladder rung could ever exercise a large window.
     MUST be read BEFORE the munmap below: `eh` is unmapped there, so parsing after it is
     a use-after-unmap. The identical mistake was made in libcapstone and is recorded in
     that file; I repeated it here and it wedged the board on BOTH rungs, including a
     control that had passed an hour earlier. A garbage globals_off makes the monitor
     split gp out of bounds, which faults in M-mode -- a silent, total hang. */
  res->globals_off = 0;
  if (eh->e_shoff != 0 && eh->e_shnum != 0 && eh->e_shstrndx < eh->e_shnum) {
    struct Shdr *sh = (struct Shdr *)((char *)eh + eh->e_shoff);
    const char *strtab = (const char *)eh + sh[eh->e_shstrndx].sh_offset;
    int s;
    for (s = 0; s < eh->e_shnum; s++) {
      const char *nm = strtab + sh[s].sh_name;
      if (nm[0]=='.' && nm[1]=='c' && nm[2]=='a' && nm[3]=='p' &&
          streq_(nm, ".capstone_gp_initdesc")) {
        if (sh[s].sh_addr >= lstart) res->globals_off = sh[s].sh_addr - lstart;
        break;
      }
    }
  }
  sys_munmap(eh, (ulong)fsize);
  sys_close(fd);
  res->map_base = img; res->map_len = image_size;
  res->code_start = (ulong)img; res->code_len = image_size;
  res->entry_offset = entry - lstart;

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
  /* Packed into entry_offset's high 32 bits: the kernel module forwards this word
     untouched, so no ioctl struct change is needed in two submodule trees. The monitor
     unpacks with two 16-bit shifts (a single >>32 is evaluated at 32 bits by capstone-c).
     Entry offsets are far below 2^32, so the packing is unambiguous. */
  a.entry_offset = c.entry_offset | (c.globals_off << 32);
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
  struct ioctl_region_query_args q;
  memset_(&q, 0, sizeof(q));
  q.region_id = id;
  sys_ioctl(dev_fd, IOCTL_REGION_QUERY, &q);
  if (q.len == 0) return 0;
  void *p = sys_mmap(0, len, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd,
                     (long)q.mmap_offset);
  return mmap_ok(p) ? p : 0;
}

static void shared_region_annotated(dom_id_t dom, region_id_t reg,
                                    ulong perm, ulong rev) {
  struct ioctl_region_share_annotated_args a;
  memset_(&a, 0, sizeof(a));
  a.dom_id = dom; a.region_id = reg; a.annotation_perm = perm; a.annotation_rev = rev;
  sys_ioctl(dev_fd, IOCTL_REGION_SHARE_ANNOTATED, &a);
}

static int run(const char *name, const char *dom_path) {
  dev_fd = sys_open(CAPSTONE_DEV_PATH_STR, O_NONBLOCK | O_RDWR);
  if (dev_fd < 0) { puts_(TAG ": open /dev/capstone failed\n"); return 1; }

  /* PHASE MARKER, one write, before the ioctl. Without it a wedge INSIDE domain
     creation and a wedge just AFTER it are indistinguishable on the console: both
     end the capture with no further output. That ambiguity cost the SQLite bring-up
     several board runs (see the phase markers added to sqlite_host.c), and the 2 MiB
     rungs are exactly where it matters, since their create_domain does an M-mode
     copy three orders of magnitude longer than a small rung's.
     The text is chosen so its FIRST 16 CHARACTERS are unique against the line below
     ("ladder-perf: A/p" vs "ladder-perf: <rung>"): the observed board failure
     truncates output at a 16-byte tty flush boundary, so a marker that is only
     distinguishable past character 16 does not survive the failure it exists to
     observe. */
  puts_(TAG ": A/pre-create-dom\n");
  dom_id_t dom = create_dom(dom_path);
  if ((long)dom < 0) { puts_(TAG ": create_dom failed\n"); return 1; }
  puts_(TAG ": "); puts_(name); puts_(" domain ID = "); putu_(dom); puts_("\n");

  region_id_t region_id = create_region(REGION_SIZE);
  unsigned char *results = map_region(region_id, REGION_SIZE);
  if (!results) { puts_(TAG ": map region failed\n"); return 1; }
  memset_(results, 0, REGION_SIZE);

  /* The share IS the entry: domain_main runs the rung and writes res[0..2]. */
  shared_region_annotated(dom, region_id, ANNOT_PERM_INOUT, REV_SHARED);

  /* res[64]/res[65]: retired-instruction delta and the minstret phase marker
     (ladder_perf_domain.h). They sit above the debug window on purpose. instret
     is what separates "the capability ABI retires more instructions" from "the
     same instructions cost more cycles"; phase says whether minstret was even
     readable, so a domain that faults reading it does not look like a generic
     silent failure. Diagnostic rungs have their own domain_main and write
     neither, so both print only when set. */
  const ulong *r = (const ulong *)results;
  ulong retval = r[0], cycles = r[1], ran = r[2];
  ulong instret = r[64], phase = r[65];
  puts_(TAG ": RESULT "); puts_(name);
  puts_(" retval="); putu_(retval);
  puts_(" cycles="); putu_(cycles);
  puts_(" ran="); putu_(ran);
  if (phase) { puts_(" instret="); putu_(instret);
               puts_(" phase="); putu_(phase); }
  puts_("\n");

  /* Diagnostic rungs (gp_diag) additionally write one RAW value per probe into
     res[3..]; perf rungs leave those slots zero. Printing them costs a perf rung
     nothing and saves a whole power-cycle when diagnosing. */
  {
    int any = 0;
    for (int i = 0; i < LADDER_DBG_SLOTS; i++)
      if (r[3 + i]) { any = 1; break; }
    if (any) {
      puts_(TAG ": DEBUG "); puts_(name);
      for (int i = 0; i < LADDER_DBG_SLOTS; i++) {
        puts_(" dbg"); putu_((ulong)i); puts_("=");
        putu_(r[3 + i]);
      }
      puts_("\n");
    }
  }
  return 0;
}

/* ------------------------------------------------------------------ *
 *  Entry: parse argc/argv off the stack, no libc runtime.
 *  usage: ladder_perf_ctl <name> <rung.dom>
 * ------------------------------------------------------------------ */
void _start_c(long *sp) {
  long argc = sp[0];
  char **argv = (char **)(sp + 1);
  if (argc < 3) {
    puts_("usage: ladder_perf_ctl <name> <rung.dom>\n");
    sys_exit(2);
  }
  int rc = run(argv[1], argv[2]);
  sys_exit(rc);
}

__asm__(
  ".section .text\n"
  ".globl _start\n"
  "_start:\n"
  "  .option push\n"
  "  .option norelax\n"
  "  lla gp, __global_pointer$\n"
  "  .option pop\n"
  "  mv a0, sp\n"
  "  andi sp, sp, -16\n"
  "  call _start_c\n"
  "  ebreak\n"
);
