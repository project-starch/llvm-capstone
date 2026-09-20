/* Report a program's fault from OUTSIDE it.
 *
 *     supervise PROGRAM [ARGS...]
 *
 * The corpus used to ask the program under test what happened to it: a handler
 * inside it printed the signal, the code and the trap PC, and even its own
 * verdict on whether the PC was the right one. That is a self-report, and a
 * self-report is the weakest kind of evidence an instrument can produce -- a
 * bug in the printing, a lost buffer or a wrong comparison inside the program
 * all become a wrong measurement.
 *
 * This observes instead. It forks, traces the child, and reads what the KERNEL
 * says: PT_LWPINFO gives the signal, si_code and si_addr; PT_GETCAPREGS gives
 * the capability PC. The program under test prints nothing and decides nothing.
 *
 * Three lines are printed, all from the parent:
 *
 *   SUPERVISE base 0x... <path>        the child's text mapping
 *   SUPERVISE expect <symbol> 0x...    that base plus the symbol's ELF value
 *   SUPERVISE fault signal=N code=N addr=0x... pc=0x...
 *
 * The expected address is resolved here rather than asked of the child: the load
 * base comes from the child's own memory map (kinfo_getvmmap), and the symbol's
 * value is read out of the target ELF. Do NOT derive the base from the exec
 * stop -- these are dynamically linked, so the exec stop is in the run-time
 * linker, not in the program. That was tried and measured wrong: an exec PC of
 * 0x401155e0 against an e_entry of 0x3628 gives nothing useful.
 *
 * The child's own exit status is passed through, so a SIGPROT still surfaces as
 * 162 the way the runner expects.
 */
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <libutil.h>
#include <machine/reg.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

/* The labelled access every case's protected arm must fault at. */
#define PROBE_SYMBOL "pyc_defect_read"

#ifdef __CHERI_PURE_CAPABILITY__
#include <cheri/cheric.h>
#define PC_OF(regs) ((unsigned long)cheri_getaddress((void *)(regs).sepcc))
#else
#define PC_OF(regs) ((unsigned long)(regs).sepc)
#endif

/* The load base of the child's main object: the LOWEST mapping of that file.
 *
 * Not the executable mapping -- that one starts at the text segment, which for
 * these binaries sits 0x3000 above the base, and adding a symbol to it
 * overshoots by exactly that. The first PT_LOAD has p_vaddr 0, so its mapping
 * is the base. Reported as 0 if it cannot be found, never guessed. */
static unsigned long text_base(pid_t pid, const char *path) {
  int count = 0;
  struct kinfo_vmentry *map = kinfo_getvmmap(pid, &count);
  unsigned long base = 0;
  if (map == NULL)
    return 0;
  /* The map carries absolute paths; the caller may have passed a relative one,
   * as the runner does with ./target. Compare resolved against resolved. */
  char resolved[PATH_MAX];
  if (realpath(path, resolved) == NULL)
    snprintf(resolved, sizeof resolved, "%s", path);
  for (int i = 0; i < count; i++) {
    if (!strcmp(map[i].kve_path, resolved)) {
      unsigned long start = (unsigned long)map[i].kve_start;
      if (base == 0 || start < base)
        base = start;
    }
  }
  free(map);
  return base;
}

/* A symbol's value in the target ELF. Zero if absent, never guessed. */
static unsigned long elf_symbol(const char *path, const char *want) {
  FILE *f = fopen(path, "rb");
  if (f == NULL)
    return 0;
  Elf64_Ehdr eh;
  unsigned long value = 0;
  if (fread(&eh, sizeof eh, 1, f) != 1 || memcmp(eh.e_ident, ELFMAG, SELFMAG))
    goto done;
  for (unsigned i = 0; i < eh.e_shnum; i++) {
    Elf64_Shdr sh, str;
    if (fseek(f, (long)(eh.e_shoff + (unsigned long)i * eh.e_shentsize), SEEK_SET) ||
        fread(&sh, sizeof sh, 1, f) != 1 || sh.sh_type != SHT_SYMTAB)
      continue;
    if (fseek(f, (long)(eh.e_shoff + (unsigned long)sh.sh_link * eh.e_shentsize),
              SEEK_SET) || fread(&str, sizeof str, 1, f) != 1)
      continue;
    char *names = malloc(str.sh_size);
    Elf64_Sym *syms = malloc(sh.sh_size);
    if (names == NULL || syms == NULL) {
      free(names); free(syms); continue;
    }
    if (!fseek(f, (long)str.sh_offset, SEEK_SET) &&
        fread(names, str.sh_size, 1, f) == 1 &&
        !fseek(f, (long)sh.sh_offset, SEEK_SET) &&
        fread(syms, sh.sh_size, 1, f) == 1) {
      for (unsigned long k = 0; k < sh.sh_size / sizeof *syms; k++)
        if (syms[k].st_name < str.sh_size &&
            !strcmp(names + syms[k].st_name, want)) {
          value = (unsigned long)syms[k].st_value;
          break;
        }
    }
    free(names);
    free(syms);
    if (value)
      break;
  }
done:
  fclose(f);
  return value;
}

static int trace(int request, pid_t pid, void *addr, int data) {
  errno = 0;
  return ptrace(request, pid, (caddr_t)addr, data);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fputs("usage: supervise PROGRAM [ARGS...]\n", stderr);
    return 2;
  }
  pid_t child = fork();
  if (child < 0) {
    perror("fork");
    return 2;
  }
  if (child == 0) {
    if (trace(PT_TRACE_ME, 0, NULL, 0)) {
      perror("PT_TRACE_ME");
      _exit(2);
    }
    execv(argv[1], argv + 1);
    perror("execv");
    _exit(2);
  }

  int reported = 0, status = 0;
  for (;;) {
    if (waitpid(child, &status, 0) < 0) {
      perror("waitpid");
      return 2;
    }
    if (!WIFSTOPPED(status))
      break;

    int signo = WSTOPSIG(status);
#ifdef __CHERI_PURE_CAPABILITY__
    struct capreg regs;
    int have_regs = trace(PT_GETCAPREGS, child, &regs, 0) == 0;
#else
    struct reg regs;
    int have_regs = trace(PT_GETREGS, child, &regs, 0) == 0;
#endif

    if (signo == SIGTRAP && !reported) {
      /* The exec stop: the program is mapped, so its base and the address the
       * probe will sit at can both be resolved without asking it anything. */
      unsigned long base = text_base(child, argv[1]);
      unsigned long symbol = elf_symbol(argv[1], PROBE_SYMBOL);
      char shown[PATH_MAX];
      if (realpath(argv[1], shown) == NULL)
        snprintf(shown, sizeof shown, "%s", argv[1]);
      printf("SUPERVISE base 0x%lx %s\n", base, shown);
      if (base && symbol)
        printf("SUPERVISE expect %s 0x%lx\n", PROBE_SYMBOL, base + symbol);
      else
        printf("SUPERVISE expect %s unavailable\n", PROBE_SYMBOL);
      fflush(stdout);
      reported = 1;
      trace(PT_CONTINUE, child, (void *)1, 0);
      continue;
    }

    struct ptrace_lwpinfo info;
    memset(&info, 0, sizeof info);
    int have_info = trace(PT_LWPINFO, child, &info, sizeof info) == 0;
    if (have_info && (info.pl_flags & PL_FLAG_SI))
      printf("SUPERVISE fault signal=%d code=%d addr=0x%lx pc=0x%lx\n",
             info.pl_siginfo.si_signo, info.pl_siginfo.si_code,
             (unsigned long)(__cheri_addr long)info.pl_siginfo.si_addr,
             have_regs ? PC_OF(regs) : 0UL);
    else
      printf("SUPERVISE fault signal=%d code=unavailable addr=unavailable "
             "pc=0x%lx\n",
             signo, have_regs ? PC_OF(regs) : 0UL);
    fflush(stdout);
    /* Let the signal through so the child dies its own death. */
    trace(PT_CONTINUE, child, (void *)1, signo);
  }

  if (WIFSIGNALED(status)) {
    printf("SUPERVISE exit signalled=%d\n", WTERMSIG(status));
    fflush(stdout);
    return 128 + WTERMSIG(status);
  }
  printf("SUPERVISE exit status=%d\n", WEXITSTATUS(status));
  fflush(stdout);
  return WEXITSTATUS(status);
}
