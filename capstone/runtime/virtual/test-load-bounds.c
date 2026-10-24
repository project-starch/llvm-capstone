/* load() reads the program and section headers again from the page-cache mapping after
 * capstone_application_validate. A change below the VFS in between (a loop device's backing
 * file, a raw block device: the residual the README names) may make those fields garbage. This
 * test stands in for that change: the validation is stubbed out, load() is given images whose
 * fields are each wrong in one way, and every case must come back ENOEXEC (or, for the control,
 * load) without a crash. Built against the exec.c named by EXEC_C, so the previous commit's
 * load() can be run through the same cases: it must fail on them, which shows the cases can
 * fail. Host test: it needs no module and no capability hardware. */
#define _GNU_SOURCE
#define main vexec_main
#define capstone_application_validate test_validate_stub
#include EXEC_C
#undef main
#undef capstone_application_validate
int test_validate_stub(const void *data, size_t size, struct capstone_application_descriptor_v2 *desc)
{
    (void)data; (void)size; (void)desc;
    return 0;
}
#include <sys/wait.h>

enum { PHOFF, PHNUM, SHOFF, SHNUM, SHSTRNDX, NAMES_OFF, NAMES_LEN, SH_NAME, SEC_OFF, FILESZ, SEG_OFF,
       ENTRY, NCASES };
static const char *const NAME[NCASES] = {
    "e_phoff beyond the file", "e_phnum larger than the file holds", "e_shoff beyond the file",
    "e_shnum larger than the file holds", "e_shstrndx out of range", "string table offset beyond the file",
    "string table longer than the file", "sh_name beyond the string table",
    "section data offset beyond the file", "p_filesz beyond the reservation",
    "p_offset beyond the file", "e_entry outside the reservation" };

#define NSEC 4
static size_t build(unsigned char *buf, int mutate)
{
    static const char strtab[] = "\0.capstone_virtual\0.shstrtab";
    Elf64_Ehdr *h = (void *)buf;
    Elf64_Phdr *ph = (void *)(buf + sizeof *h);
    unsigned data_at = sizeof *h + sizeof *ph, str_at = data_at + 8, sh_at = (str_at + sizeof strtab + 7) & ~7u;
    Elf64_Shdr *sh = (void *)(buf + sh_at);
    uint64_t magic = CV_IMAGE_MAGIC_V3;
    memset(buf, 0, sh_at + NSEC * sizeof *sh);
    memcpy(h->e_ident, ELFMAG, SELFMAG);
    h->e_ident[EI_CLASS] = ELFCLASS64; h->e_ident[EI_DATA] = ELFDATA2LSB; h->e_ident[EI_VERSION] = EV_CURRENT;
    h->e_type = ET_EXEC; h->e_machine = EM_RISCV; h->e_version = EV_CURRENT;
    h->e_phoff = sizeof *h; h->e_phentsize = sizeof *ph; h->e_phnum = 1;
    h->e_shoff = sh_at; h->e_shentsize = sizeof *sh; h->e_shnum = NSEC; h->e_shstrndx = 2;
    h->e_entry = 0x1000;
    ph->p_type = PT_LOAD; ph->p_flags = PF_R | PF_W | PF_X; ph->p_offset = 0; ph->p_vaddr = 0x1000;
    ph->p_filesz = sh_at; ph->p_memsz = 0x4000;
    memcpy(buf + data_at, &magic, 8);
    memcpy(buf + str_at, strtab, sizeof strtab);
    sh[1].sh_type = SHT_PROGBITS; sh[1].sh_name = 1; sh[1].sh_offset = data_at; sh[1].sh_size = 8;
    sh[2].sh_type = SHT_STRTAB; sh[2].sh_name = 19; sh[2].sh_offset = str_at; sh[2].sh_size = sizeof strtab;
    switch (mutate) {
    case PHOFF: h->e_phoff = UINT64_C(0x7fffffffffff); break;
    case PHNUM: h->e_phnum = 0xffff; break;
    case SHOFF: h->e_shoff = UINT64_C(0x7fffffffffff); break;
    case SHNUM: h->e_shnum = 0xffff; break;
    case SHSTRNDX: h->e_shstrndx = 0xfff0; break;
    case NAMES_OFF: sh[2].sh_offset = UINT64_C(0x7fffffffffff); break;
    case NAMES_LEN: sh[2].sh_size = UINT64_C(0x7fffffffffff); break;
    case SH_NAME: sh[1].sh_name = 0x7ffffff0; break;
    case SEC_OFF: sh[1].sh_offset = UINT64_C(0x7fffffffffff); break;
    case FILESZ: ph->p_filesz = UINT64_C(0x7fffffffffff); break;
    case SEG_OFF: ph->p_offset = UINT64_C(0x7fffffffffff); break;
    case ENTRY: h->e_entry = UINT64_C(0x7fffffffffff); break;
    default: break;
    }
    return sh_at + NSEC * sizeof *sh;
}

/* Run one case in a child: exit status 0 = load refused it (ENOEXEC), 1 = load accepted it,
 * 2 = refused with another errno, a signal = the process died. */
static int run(int mutate)
{
    unsigned char buf[4096];
    size_t n = build(buf, mutate);
    char path[] = "/tmp/load-bounds-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0 || write(fd, buf, n) != (ssize_t)n) return 99;
    unlink(path);
    pid_t pid = fork();
    if (!pid) {
        void *image; size_t bytes, stack; uintptr_t entry;
        errno = 0;
        int rc = load(fd, NULL, &image, &bytes, &entry, &stack);
        _exit(rc == 0 ? 1 : errno == ENOEXEC ? 0 : 2);
    }
    int status; waitpid(pid, &status, 0);
    close(fd);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 100 + WTERMSIG(status);
}

int main(void)
{
    int bad = 0;
    int control = run(-1);
    printf("%-44s %s\n", "control (unmodified image)", control == 1 ? "loads" : "DOES NOT LOAD");
    if (control != 1) bad++;
    for (int i = 0; i < NCASES; i++) {
        int r = run(i);
        printf("%-44s %s\n", NAME[i], r == 0 ? "refused (ENOEXEC)" : r == 1 ? "ACCEPTED" :
               r == 2 ? "refused, other errno" : r == 99 ? "test error" : "CRASHED (signal)");
        if (r != 0) bad++;
    }
    printf("%s: %d of %d cases not refused cleanly\n", bad ? "FAIL" : "PASS", bad, NCASES + 1);
    return bad != 0;
}
