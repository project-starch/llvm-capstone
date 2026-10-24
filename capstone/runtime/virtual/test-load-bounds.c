/* load() reads the program and section headers again from the page-cache mapping after
 * capstone_application_validate (the layout of the two-segment image is read from them). A change below the VFS in between (a loop device's backing
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

enum { PHOFF, PHNUM, SHOFF, SHNUM, SHSTRNDX, NAMES_OFF, NAMES_LEN, SH_NAME, SEC_OFF,
       TEXT_FILESZ, TEXT_OFF, DATA_FILESZ, DATA_OFF, DATA_MEMSZ, ENTRY, DOMREQ_OFF, NCASES };
static const char *const NAME[NCASES] = {
    "e_phoff beyond the file", "e_phnum larger than the file holds", "e_shoff beyond the file",
    "e_shnum larger than the file holds", "e_shstrndx out of range", "string table offset beyond the file",
    "string table longer than the file", "sh_name beyond the string table",
    "section data offset beyond the file", "text p_filesz beyond the file",
    "text p_offset beyond the file", "data p_filesz beyond the file",
    "data p_offset beyond the file", "data p_memsz wraps the sum", "e_entry outside the text",
    "stack request section beyond the file" };

#define NSEC 4
#define FILE_BYTES 8192
/* The stack the control image asks for (.capstone_domreq): rounded() makes it a power of two. */
#define REQ_STACK (3u << 19)
/* The image as link.ld lays it out, in miniature: page 0 is the read+execute text (headers,
 * strings and the magic live in it), page 1 starts the read+write data. */
static size_t build(unsigned char *buf, int mutate)
{
    static const char strtab[] = "\0.capstone_virtual\0.shstrtab\0.capstone_domreq";
    Elf64_Ehdr *h = (void *)buf;
    Elf64_Phdr *ph = (void *)(buf + sizeof *h);
    unsigned data_at = sizeof *h + 2 * sizeof *ph, str_at = data_at + 8,
             req_at = (str_at + sizeof strtab + 7) & ~7u, sh_at = req_at + 24;
    Elf64_Shdr *sh = (void *)(buf + sh_at);
    uint64_t magic = CV_IMAGE_MAGIC, req[3] = {0, 0, REQ_STACK};
    memset(buf, 0, FILE_BYTES);
    memcpy(h->e_ident, ELFMAG, SELFMAG);
    h->e_ident[EI_CLASS] = ELFCLASS64; h->e_ident[EI_DATA] = ELFDATA2LSB; h->e_ident[EI_VERSION] = EV_CURRENT;
    h->e_type = ET_EXEC; h->e_machine = EM_RISCV; h->e_version = EV_CURRENT;
    h->e_phoff = sizeof *h; h->e_phentsize = sizeof *ph; h->e_phnum = 2;
    h->e_shoff = sh_at; h->e_shentsize = sizeof *sh; h->e_shnum = NSEC; h->e_shstrndx = 2;
    h->e_entry = 0x1000;
    ph[0].p_type = PT_LOAD; ph[0].p_flags = PF_R | PF_X; ph[0].p_offset = 0; ph[0].p_vaddr = 0x1000;
    ph[0].p_filesz = ph[0].p_memsz = 0x1000;
    ph[1].p_type = PT_LOAD; ph[1].p_flags = PF_R | PF_W; ph[1].p_offset = 0x1000; ph[1].p_vaddr = 0x2000;
    ph[1].p_filesz = 64; ph[1].p_memsz = 0x2000;
    memcpy(buf + data_at, &magic, 8);
    memcpy(buf + str_at, strtab, sizeof strtab);
    memcpy(buf + req_at, req, sizeof req);
    sh[1].sh_type = SHT_PROGBITS; sh[1].sh_name = 1; sh[1].sh_offset = data_at; sh[1].sh_size = 8;
    sh[2].sh_type = SHT_STRTAB; sh[2].sh_name = 19; sh[2].sh_offset = str_at; sh[2].sh_size = sizeof strtab;
    sh[3].sh_type = SHT_PROGBITS; sh[3].sh_name = 29; sh[3].sh_offset = req_at; sh[3].sh_size = 24;
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
    /* three pages of text in a two-page file; the data segment follows them, so that only
     * the file-length check can refuse this */
    case TEXT_FILESZ: ph[0].p_filesz = ph[0].p_memsz = 0x3000;
                      ph[1].p_vaddr = ph[0].p_vaddr + ph[0].p_memsz; break;
    case TEXT_OFF: ph[0].p_offset = UINT64_C(0x7fffffffe000); break;
    case DATA_FILESZ: ph[1].p_filesz = ph[1].p_memsz = 0x3000; break; /* from page 1 of a 2-page file */
    case DATA_OFF: ph[1].p_offset = UINT64_C(0x7fffffffe000); break;
    case DATA_MEMSZ: ph[1].p_memsz = UINT64_C(0xfffffffffffff000); break;
    case ENTRY: h->e_entry = UINT64_C(0x7fffffffffff); break;
    case DOMREQ_OFF: sh[3].sh_offset = UINT64_C(0x7fffffffffff); break;
    default: break;
    }
    return FILE_BYTES;
}

/* Run one case in a child: exit status 0 = load refused it (ENOEXEC), 1 = load accepted it,
 * 2 = refused with another errno, a signal = the process died. */
static int run(int mutate)
{
    unsigned char buf[FILE_BYTES];
    size_t n = build(buf, mutate);
    char path[] = "/tmp/load-bounds-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0 || write(fd, buf, n) != (ssize_t)n) return 99;
    unlink(path);
    pid_t pid = fork();
    if (!pid) {
        void *image; size_t bytes, rx, stack; uintptr_t entry;
        errno = 0;
        int rc = load(fd, NULL, &image, &bytes, &rx, &entry, &stack);
        /* A loaded image carries the stack it asked for, or the default when the request
         * section lies outside the file (ignored, not read). */
        if (rc == 0 && stack != (mutate == DOMREQ_OFF ? (size_t)1 << 20 : rounded(REQ_STACK))) _exit(3);
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
    printf("%-44s %s\n", "control (unmodified image)",
           control == 1 ? "loads, asked-for stack" : control == 3 ? "LOADS WITH THE WRONG STACK" : "DOES NOT LOAD");
    if (control != 1) bad++;
    for (int i = 0; i < NCASES; i++) {
        int r = run(i);
        /* A request section outside the file is not read: the image loads with the default stack. */
        int want = i == DOMREQ_OFF ? 1 : 0;
        printf("%-44s %s\n", NAME[i], r == 0 ? "refused (ENOEXEC)" : r == 1 ? (want ? "loads, default stack" : "ACCEPTED") :
               r == 2 ? "refused, other errno" : r == 3 ? "WRONG STACK" : r == 99 ? "test error" : "CRASHED (signal)");
        if (r != want) bad++;
    }
    printf("%s: %d of %d cases not refused cleanly\n", bad ? "FAIL" : "PASS", bad, NCASES + 1);
    return bad != 0;
}
