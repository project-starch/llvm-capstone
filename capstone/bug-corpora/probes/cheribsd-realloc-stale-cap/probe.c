/* Is the OLD capability after realloc() stale but still accessible on CheriBSD?
 *
 * C17 7.22.3.5 ends the old object's lifetime on every successful realloc, whether or not the
 * returned address is the same. This probe asks what CheriBSD's libc actually does with the old
 * capability in each shape, under BOTH its layers:
 *
 *   revocation on  (the default: the malloc revocation shim in front of jemalloc)
 *   revocation off (_RUNTIME_REVOCATION_DISABLE=1: CHERI-patched jemalloc alone)
 *
 * Each scenario runs in a forked child. The child prints the two capabilities, whether the
 * address stayed the same, the old capability's tag at the moment of the access, and then
 * accesses memory THROUGH THE OLD CAPABILITY at a labelled instruction. The parent reports whether
 * the child completed or died of SIGPROT. Every line is the reading; nothing is asserted.
 *
 * Three kinds of scenario:
 *   READ        read old[index]
 *   TWO_OWNERS  the FFmpeg diracdec shape (upstream fix 8a4ea9644833): the realloc's result is
 *               stored in a second field, the first keeps the old pointer and is still used. The
 *               child stores a marker through the OLD capability, then reads the same offset
 *               through the NEW one: an equal byte means the stale write landed in the live
 *               object (same address) or was absorbed by the dead block (moved).
 *   SHRINK_TAIL a large block shrunk in place, then up to 16 fresh allocations: if one lands
 *               inside the OLD capability's range, the child writes a marker through the old
 *               capability at that address and reads it back through the fresh object.
 *
 * Sizes are chosen against the two layers' rules (contrib/jemalloc and mrs.c in CheriBSD):
 *   jemalloc small: in place only when the size class is unchanged (256 -> 250 yes, 256 -> 200 no)
 *   jemalloc large: large_ralloc_no_move shrinks in place, trimming the extent tail
 *   shim: same pointer when the request fits and reclaims at most half; otherwise malloc+copy+free
 */
#include <cheri/cheric.h>
#include <malloc_np.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

enum kind { READ, TWO_OWNERS, SHRINK_TAIL };
struct scenario {
  const char *name;
  enum kind kind;
  size_t first, second; /* malloc(first), realloc(.., second) */
  size_t index;         /* byte accessed through the OLD capability */
  int sweep;            /* force a revocation sweep before the access */
};

static const struct scenario SCENARIOS[] = {
    {"shrink-small-same-class", READ, 256, 250, 255, 0},
    {"shrink-small-other-class", READ, 256, 200, 0, 0},
    {"grow-small-within-class", READ, 100, 112, 0, 0},
    {"shrink-large-in-place", READ, 1u << 20, 600u * 1024, (1u << 20) - 1, 0},
    {"grow-moved-no-sweep", READ, 64, 1u << 16, 0, 0},
    {"grow-moved-sweep", READ, 64, 1u << 16, 0, 1},
    {"shrink-moved-sweep", READ, 256, 64, 0, 1},
    {"two-owners-in-place", TWO_OWNERS, 256, 250, 0, 0},
    {"two-owners-moved", TWO_OWNERS, 64, 1u << 16, 0, 0},
    {"shrink-tail-reissued", SHRINK_TAIL, 1u << 20, 600u * 1024, 0, 0},
    /* Reconstructions of two historical upstream bugs of this exact shape. Both fixes
     * predate our pins (mruby eb398971b is in 4.0.0-rc2; Perl 4547997058, 2014, is in
     * 5.36.3), so neither is a live release-differential case: the only way to measure the
     * defect is to reconstruct its realloc shape here. Sizes are chosen so the realloc keeps
     * the same address on the layer named; the real defects do not depend on the exact sizes.
     *
     *   mruby Array#sort! use-after-realloc, fix eb398971b (#6649): sort_cmp captured the
     *   backing-store base and the length; a comparison block that shrinks the array makes
     *   ary_shrink_capa() realloc the store smaller, and the old guard was a pointer-equality
     *   test, so a same-address shrink was missed and the sort read the stale tail. 512 -> 256 B
     *   hits the shim's shrink shortcut (reclaims exactly half), so the block keeps its address
     *   under the default revocation-on layer; index 504 lies in the trimmed tail. */
    {"mruby-sort-bang-shrink", READ, 512, 256, 504, 0},
    /*   Perl Gv_AMupdate stale HvAUX, fix 4547997058 (RT #121362): aux = HvAUX(stash) is an
     *   interior pointer into the tail of the bucket array; adding a method triggers S_hsplit,
     *   whose Renew() can grow the array in place, after which aux->xhv_aux_flags |= ... writes
     *   through the stale interior pointer into live buckets. malloc(200) already yields a 224 B
     *   block, so realloc(.., 224) keeps the same address under BOTH layers: the shim measures
     *   old_size by the capability length (224), so size <= old_size fires its shortcut, and
     *   jemalloc leaves it in place inside one small size class. index 192 is the interior aux
     *   slot; the TWO_OWNERS oracle (write old, read new) sees the stale write land in the live
     *   object, so this is a silent live-object corruption on CheriBSD under revocation on. */
    {"perl-hsplit-aux-grow", TWO_OWNERS, 200, 224, 192, 0},
};

__attribute__((noinline)) static unsigned read_old(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl realloc_probe_read\nrealloc_probe_read:\nclbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "C"(p)
                   : "memory");
  return (unsigned)value;
}

__attribute__((noinline)) static void write_old(volatile unsigned char *p, unsigned char v) {
  __asm__ volatile(".globl realloc_probe_write\nrealloc_probe_write:\ncsb %0, 0(%1)\n"
                   :
                   : "r"(v), "C"(p)
                   : "memory");
}

static void describe(const struct scenario *s, unsigned revocation, void *keep, void *new) {
  printf("SCEN %s revocation=%u old=%#p new=%#p same_addr=%d old_len=%zu new_len=%zu "
         "old_tag=%u index=%zu\n",
         s->name, revocation, keep, new, cheri_getaddress(keep) == cheri_getaddress(new),
         cheri_getlen(keep), cheri_getlen(new), (unsigned)cheri_gettag(keep), s->index);
  fflush(stdout);
}

static void child(const struct scenario *s, unsigned revocation) {
  unsigned char *old = malloc(s->first);
  if (!old) _exit(2);
  memset(old, 0x5a, s->first);
  unsigned char *volatile keep = old; /* the stale copy a program would hold */
  unsigned char *new = realloc(old, s->second);
  if (!new) _exit(2);
  if (s->sweep && revocation) malloc_revoke_quarantine_force_flush();
  describe(s, revocation, (void *)keep, new);
  switch (s->kind) {
  case READ: {
    unsigned v = read_old(keep + s->index);
    printf("ACCESS %s completed value=0x%02x\n", s->name, v);
    break;
  }
  case TWO_OWNERS: {
    write_old(keep + s->index, 0xa5);
    unsigned seen = new[s->index];
    printf("ACCESS %s completed stale_write_seen_through_new=%d value=0x%02x\n", s->name,
           seen == 0xa5, seen);
    break;
  }
  case SHRINK_TAIL: {
    unsigned char *fresh[16];
    int hit = -1;
    for (int i = 0; i < 16 && hit < 0; i++) {
      fresh[i] = malloc(200u * 1024);
      if (!fresh[i]) break;
      ptraddr_t a = cheri_getaddress(fresh[i]);
      if (a >= cheri_getbase((void *)keep) && a < cheri_getbase((void *)keep) + cheri_getlen((void *)keep))
        hit = i;
    }
    if (hit < 0) {
      printf("ACCESS %s no fresh allocation landed inside the old capability's range\n", s->name);
      break;
    }
    size_t off = cheri_getaddress(fresh[hit]) - cheri_getaddress((void *)keep);
    printf("TAIL %s fresh=%#p lies at old+%zu (new object ends at %zu)\n", s->name, fresh[hit], off,
           cheri_getlen(new));
    fflush(stdout);
    write_old(keep + off, 0xa5);
    unsigned seen = fresh[hit][0];
    printf("ACCESS %s completed stale_write_seen_through_fresh=%d value=0x%02x\n", s->name,
           seen == 0xa5, seen);
    break;
  }
  }
  fflush(stdout);
  free(new);
  _exit(0);
}

static void run_all(unsigned revocation) {
  for (size_t i = 0; i < sizeof SCENARIOS / sizeof SCENARIOS[0]; i++) {
    const struct scenario *s = &SCENARIOS[i];
    pid_t pid = fork();
    if (pid < 0) exit(2);
    if (pid == 0) child(s, revocation);
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFSIGNALED(st))
      printf("RESULT %s revocation=%u outcome=signal%d%s\n", s->name, revocation, WTERMSIG(st),
             WTERMSIG(st) == SIGPROT ? "(SIGPROT)" : "");
    else
      printf("RESULT %s revocation=%u outcome=exit%d\n", s->name, revocation, WEXITSTATUS(st));
    fflush(stdout);
  }
}

int main(int argc, char **argv) {
  unsigned revocation = (unsigned)malloc_revoke_enabled();
  run_all(revocation);
  if (argc == 1 && revocation) {
    /* second arm: the same binary with the shim's quarantine disabled, jemalloc alone */
    pid_t pid = fork();
    if (pid == 0) {
      setenv("_RUNTIME_REVOCATION_DISABLE", "1", 1);
      execl(argv[0], argv[0], "norev", (char *)NULL);
      _exit(2);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    printf("ARM norev exit=%d\n", WIFEXITED(st) ? WEXITSTATUS(st) : -1);
  }
  if (argc == 1) puts("PROBE_DONE");
  return 0;
}
