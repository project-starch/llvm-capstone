/* SPDX-License-Identifier: GPL-2.0-only */
/* The .dom loader: it parses a protected program's image, maps it with its stack, startup,
 * metadata and exchange regions, creates the context and derives one block per region from R.
 * Root only. The protected entry does not exist yet (plan v5 steps 3 and 4), so the loader
 * refuses after it has built everything. Past the point of no return that kills the task. */
#include <linux/binfmts.h>
#include <linux/elf.h>
#include <linux/mman.h>
#include <linux/uio.h>
#include <linux/fs_struct.h>
#include <linux/path.h>
#include <linux/cred.h>
#include <linux/pid_namespace.h>
#include <linux/ptrace.h>
#include <linux/security.h>
#include <asm/processor.h>
#include <asm/cacheflush.h>
#include "capstone/delegate.h"

/* Read-only attribution record: the image base of the last load. */
static unsigned long dom_last_base;
module_param(dom_last_base, ulong, 0400);
static bool dom_binfmt;
module_param(dom_binfmt, bool, 0444);
MODULE_PARM_DESC(dom_binfmt, "required root-only native .dom exec admission");
#define DOM_LAUNCH_BYTES 65536UL
#define DOM_META_BYTES 16384UL
#define DOM_MAX_STRINGS 1024
#define DOM_APP_MAGIC 0x315050414e4f5043ULL
#define DOM_REQ_MAGIC 0x5145524d4f445043ULL
#define DOM_HELLO 0xc0de0001ULL

struct dom_image {
    Elf64_Phdr text, data;
    u64 entry, bytes, stack, exchange;
};
struct dom_app { u64 magic, version, flags, launch, heap, exchange, contexts; };
struct dom_launch_header {
    u8 magic[8];
    u32 version, bytes, argc, envc, cwd, stdio_mask, pid, ppid;
    u32 uid, euid, gid, egid;
    u64 realtime_ns, monotonic_ns, ticks, ticks_per_second;
};
_Static_assert(sizeof(struct dom_launch_header) == 88, "launch ABI");
_Static_assert(sizeof(struct capstone_delegate_entry) == 96, "delegate ABI");

static int dom_read(struct file *f, void *buf, size_t n, u64 at)
{
    loff_t pos = at;
    ssize_t got = kernel_read(f, buf, n, &pos);
    return got == n ? 0 : got < 0 ? got : -ENOEXEC;
}
static bool dom_within(u64 at, u64 n, u64 size)
{ return at <= size && n <= size - at; }
static bool dom_name(const char *names, u64 len, u64 at, const char *want)
{
    size_t n = strlen(want) + 1;
    return dom_within(at, n, len) && !memcmp(names + at, want, n);
}
static int dom_parse(struct file *f, struct dom_image *im)
{
    Elf64_Ehdr h;
    Elf64_Phdr *ph = NULL;
    Elf64_Shdr *sh = NULL;
    char *names = NULL;
    struct dom_app app = {0};
    u64 size = i_size_read(file_inode(f)), magic = 0, req[3];
    unsigned i, loads = 0, markers = 0, apps = 0, reqs = 0;
    int rc = -ENOEXEC;
    if (size < sizeof(h) || size > (512UL << 20) || dom_read(f, &h, sizeof(h), 0)) return rc;
    if (memcmp(h.e_ident, ELFMAG, SELFMAG) || h.e_ident[EI_CLASS] != ELFCLASS64 ||
        h.e_ident[EI_DATA] != ELFDATA2LSB || h.e_ident[EI_VERSION] != EV_CURRENT ||
        h.e_machine != 259 || h.e_type != ET_EXEC || h.e_version != EV_CURRENT ||
        h.e_ehsize != sizeof(h) || h.e_phentsize != sizeof(*ph) ||
        h.e_shentsize != sizeof(*sh) || !h.e_phnum || h.e_phnum > 16 ||
        !h.e_shnum || h.e_shnum > 512 || h.e_shstrndx >= h.e_shnum ||
        !dom_within(h.e_phoff, h.e_phnum * sizeof(*ph), size) ||
        !dom_within(h.e_shoff, h.e_shnum * sizeof(*sh), size)) return rc;
    ph = kmalloc_array(h.e_phnum, sizeof(*ph), GFP_KERNEL);
    sh = kmalloc_array(h.e_shnum, sizeof(*sh), GFP_KERNEL);
    if (!ph || !sh) { rc = -ENOMEM; goto out; }
    if (dom_read(f, ph, h.e_phnum * sizeof(*ph), h.e_phoff) ||
        dom_read(f, sh, h.e_shnum * sizeof(*sh), h.e_shoff)) goto out;
    if (sh[h.e_shstrndx].sh_type != SHT_STRTAB || !sh[h.e_shstrndx].sh_size ||
        sh[h.e_shstrndx].sh_size > DOM_LAUNCH_BYTES ||
        !dom_within(sh[h.e_shstrndx].sh_offset, sh[h.e_shstrndx].sh_size, size)) goto out;
    names = kmalloc(sh[h.e_shstrndx].sh_size, GFP_KERNEL);
    if (!names) { rc = -ENOMEM; goto out; }
    if (dom_read(f, names, sh[h.e_shstrndx].sh_size, sh[h.e_shstrndx].sh_offset)) goto out;
    for (i = 0; i < h.e_phnum; ++i) {
        if (ph[i].p_type == PT_INTERP || ph[i].p_type == PT_DYNAMIC) goto out;
        if (ph[i].p_type != PT_LOAD) continue;
        if (loads >= 2 || !ph[i].p_memsz || ph[i].p_memsz > CV_MAX_EAGER_BYTES ||
            ph[i].p_filesz > ph[i].p_memsz || ph[i].p_vaddr > ULONG_MAX - ph[i].p_memsz ||
            !dom_within(ph[i].p_offset, ph[i].p_filesz, size)) goto out;
        if (!loads++) im->text = ph[i]; else im->data = ph[i];
    }
    if (loads != 2 || im->text.p_flags != (PF_R | PF_X) || im->data.p_flags != (PF_R | PF_W) ||
        ((im->text.p_offset | im->text.p_vaddr | im->text.p_memsz |
          im->data.p_offset | im->data.p_vaddr) & ~PAGE_MASK) ||
        im->text.p_filesz != im->text.p_memsz ||
        im->data.p_vaddr != im->text.p_vaddr + im->text.p_memsz ||
        h.e_entry < im->text.p_vaddr || h.e_entry - im->text.p_vaddr >= im->text.p_memsz)
        goto out;
    im->stack = 1UL << 20;
    for (i = 0; i < h.e_shnum; ++i) {
        Elf64_Shdr *s = sh + i;
        u64 len = sh[h.e_shstrndx].sh_size;
        if (s->sh_name >= len || !memchr(names + s->sh_name, 0, len - s->sh_name) ||
            (s->sh_type != SHT_NOBITS && !dom_within(s->sh_offset, s->sh_size, size))) goto out;
        if (dom_name(names, len, s->sh_name, ".capstone_virtual")) {
            if (markers++ || s->sh_type != SHT_PROGBITS || s->sh_size != sizeof(magic) ||
                dom_read(f, &magic, sizeof(magic), s->sh_offset)) goto out;
        } else if (dom_name(names, len, s->sh_name, ".capstone_application")) {
            if (apps++ || s->sh_type != SHT_PROGBITS || s->sh_size != sizeof(app) ||
                dom_read(f, &app, sizeof(app), s->sh_offset)) goto out;
        } else if (dom_name(names, len, s->sh_name, ".capstone_domreq")) {
            if (reqs++ || s->sh_type != SHT_PROGBITS || s->sh_size != sizeof(req) ||
                dom_read(f, req, sizeof(req), s->sh_offset) || req[0] != DOM_REQ_MAGIC ||
                req[1] < 256 || req[1] > CV_MAX_EAGER_BYTES || !req[2] || req[2] > req[1] - 256)
                goto out;
            im->stack = roundup_pow_of_two(max_t(u64, req[2], PAGE_SIZE));
        }
    }
    if (markers != 1 || apps != 1 || magic != CV_IMAGE_MAGIC || app.magic != DOM_APP_MAGIC ||
        app.version != 2 || app.flags != 3 || app.launch != DOM_LAUNCH_BYTES ||
        app.heap || app.contexts > 15 || app.exchange < PAGE_SIZE || app.exchange > (1UL << 20) ||
        (app.exchange & ~PAGE_MASK)) goto out;
    im->bytes = roundup_pow_of_two(im->text.p_memsz + PAGE_ALIGN(im->data.p_memsz));
    if (im->bytes > CV_MAX_EAGER_BYTES) goto out;
    im->entry = h.e_entry - im->text.p_vaddr;
    im->exchange = roundup_pow_of_two(app.exchange);
    rc = 0;
out:
    kfree(names); kfree(sh); kfree(ph);
    return rc;
}

/* Always private base pages; the experiment refuses THP kernels at load. */
static unsigned long dom_reserve(unsigned long bytes, unsigned long align, unsigned long prot)
{
    unsigned long raw, base, total = bytes + align - PAGE_SIZE;
    raw = vm_mmap(NULL, 0, total, prot, MAP_PRIVATE | MAP_ANONYMOUS, 0);
    if (IS_ERR_VALUE(raw)) return raw;
    base = ALIGN(raw, align);
    if (base != raw) vm_munmap(raw, base - raw);
    if (raw + total > base + bytes) vm_munmap(base + bytes, raw + total - base - bytes);
    return base;
}
static int dom_grant(struct context *c, unsigned long base, unsigned long bytes,
                     unsigned reg, unsigned long cursor, unsigned long rx)
{
    struct sublet_grant g = {.address = base, .bytes = bytes, .reg = reg,
                             .cursor = cursor, .rx_bytes = rx};
    return add(c, &g);
}

#include "delegate-dom.h"
#include "native-syscall-dom.h"

static void dom_context_destroy(struct context *c)
{
    dom_delegate_free(c);
    context_destroy(c);
}
/* The release of the context an address space carries, called by riscv_capstone_release at
 * exec and exit. */
void sublet_context_release(void *context)
{
    dom_context_destroy(context);
}
static int dom_startup(struct linux_binprm *bprm, void *block)
{
    struct dom_launch_header *h = block;
    unsigned long header_bytes = sizeof(*h);
    u32 *offsets = block + header_bytes;
    unsigned long at = bprm->p, used = header_bytes + (bprm->argc + bprm->envc) * sizeof(u32);
    struct path pwd;
    char *cwd, *tmp;
    unsigned i;
    memcpy(h->magic, "CPLAUNCH", 8);
    h->version = 2;
    h->argc = bprm->argc; h->envc = bprm->envc;
    h->pid = task_tgid_vnr(current); h->ppid = task_ppid_nr_ns(current, task_active_pid_ns(current));
    h->uid = from_kuid_munged(current_user_ns(), current_uid());
    h->euid = from_kuid_munged(current_user_ns(), current_euid());
    h->gid = from_kgid_munged(current_user_ns(), current_gid());
    h->egid = from_kgid_munged(current_user_ns(), current_egid());
    h->realtime_ns = ktime_get_real_ns(); h->monotonic_ns = ktime_get_ns();
    for (i = 0; i < 3; ++i) {
        struct file *f = fget(i);
        if (f) { h->stdio_mask |= 1U << i; fput(f); }
    }
    tmp = (char *)__get_free_page(GFP_KERNEL);
    if (!tmp) return -ENOMEM;
    get_fs_pwd(current->fs, &pwd);
    cwd = d_path(&pwd, tmp, PAGE_SIZE);
    path_put(&pwd);
    if (IS_ERR(cwd)) { free_page((unsigned long)tmp); return PTR_ERR(cwd); }
    h->cwd = used;
    i = strlen(cwd) + 1;
    memcpy(block + used, cwd, i); used += i;
    free_page((unsigned long)tmp);
    current->mm->arg_start = at;
    for (i = 0; i < bprm->argc + bprm->envc; ++i) {
        long n = strnlen_user((char __user *)at, DOM_LAUNCH_BYTES - used);
        if (n <= 0 || n > DOM_LAUNCH_BYTES - used) return -E2BIG;
        offsets[i] = used;
        if (copy_from_user(block + used, (void __user *)at, n)) return -EFAULT;
        at += n; used += n;
        if (i + 1 == bprm->argc) current->mm->arg_end = current->mm->env_start = at;
    }
    current->mm->env_end = at;
    h->bytes = used;
    return 0;
}

static struct linux_binfmt capstone_dom;
static int dom_load_binary(struct linux_binprm *bprm)
{
    struct dom_image im = {0};
    struct context *c = NULL;
    unsigned long base, stack, startup, whole, part;
    void *block = NULL, *tail = NULL;
    int rc;
    Elf64_Ehdr *h = (void *)bprm->buf;
    if (memcmp(h->e_ident, ELFMAG, SELFMAG) || h->e_machine != 259) return -ENOEXEC;
    if (!uid_eq(current_euid(), GLOBAL_ROOT_UID) || bprm->secureexec ||
        (bprm->unsafe & LSM_UNSAFE_SHARE) ||
        (file_inode(bprm->file)->i_mode & (S_ISUID | S_ISGID))) return -EACCES;
    if (!one_inode_per_file(bprm->file) || !file_inode(bprm->file)->i_nlink) return -EOPNOTSUPP;
    if (bprm->argc < 1 || bprm->argc > DOM_MAX_STRINGS ||
        bprm->envc > DOM_MAX_STRINGS - bprm->argc) return -E2BIG;
    if (current->seccomp.mode != SECCOMP_MODE_DISABLED) return -EOPNOTSUPP;
    if (current->ptrace || (read_thread_flags() & _TIF_SYSCALL_WORK))
        return -EOPNOTSUPP;
    rc = dom_parse(bprm->file, &im);
    if (rc) return rc;
    block = kvzalloc(DOM_LAUNCH_BYTES, GFP_KERNEL);
    tail = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!block || !tail) { rc = -ENOMEM; goto out; }
    /* Read the partial data page before the point of no return. */
    whole = im.data.p_filesz & PAGE_MASK; part = im.data.p_filesz - whole;
    if (part && (rc = dom_read(bprm->file, tail, part, im.data.p_offset + whole))) goto out;
    rc = begin_new_exec(bprm);
    if (rc) goto out;
    setup_new_exec(bprm);
    rc = setup_arg_pages(bprm, STACK_TOP, EXSTACK_DISABLE_X);
    if (rc) goto out;
    rc = dom_startup(bprm, block);
    if (rc) goto out;
    rc = context_create(&c);
    if (rc) goto out;
    rc = dom_resources_init(c);
    if (rc) goto out;
    rc = deny_write_access(bprm->file);
    if (rc) goto out;
    c->image = get_file(bprm->file);
    base = dom_reserve(im.bytes, im.bytes, PROT_READ | PROT_WRITE);
    if (IS_ERR_VALUE(base)) { rc = base; goto out; }
    {
        unsigned long mapped = vm_mmap(bprm->file, base, im.text.p_memsz, PROT_READ | PROT_EXEC,
                                       MAP_PRIVATE | MAP_FIXED, im.text.p_offset);
        if (IS_ERR_VALUE(mapped)) { rc = mapped; goto out; }
    }
    if (whole) {
        unsigned long mapped = vm_mmap(bprm->file, base + im.text.p_memsz, whole,
                                      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, im.data.p_offset);
        if (IS_ERR_VALUE(mapped)) { rc = mapped; goto out; }
    }
    if (part && copy_to_user((void __user *)(base + im.text.p_memsz + whole), tail, part)) {
        rc = -EFAULT; goto out;
    }
    stack = dom_reserve(im.stack, im.stack, PROT_READ | PROT_WRITE);
    startup = dom_reserve(DOM_LAUNCH_BYTES, DOM_LAUNCH_BYTES, PROT_READ | PROT_WRITE);
    c->dom_meta = dom_reserve(DOM_META_BYTES, DOM_META_BYTES, PROT_READ | PROT_WRITE);
    c->dom_exchange_bytes = im.exchange;
    c->dom_exchange = dom_reserve(im.exchange, im.exchange, PROT_READ | PROT_WRITE);
    if (IS_ERR_VALUE(stack) || IS_ERR_VALUE(startup) || IS_ERR_VALUE(c->dom_meta) ||
        IS_ERR_VALUE(c->dom_exchange)) { rc = -ENOMEM; goto out; }
    if (copy_to_user((void __user *)startup, block, ((struct dom_launch_header *)block)->bytes)) {
        rc = -EFAULT; goto out;
    }
    mutex_lock(&vm_lock);
    /* One block per region. The image's block covers text and data with its cursor at the
     * base: a data reference for the program's globals, not a code capability. The processor
     * checks no fetch, and the page table keeps the text executable and never writable. */
    rc = dom_grant(c, base, im.bytes, 3, base, im.text.p_memsz);
    if (!rc) rc = dom_grant(c, stack, im.stack, 2, stack + im.stack, 0);
    if (!rc) rc = dom_grant(c, c->dom_meta, DOM_META_BYTES, 10, c->dom_meta, 0);
    if (!rc) rc = dom_grant(c, c->dom_exchange, im.exchange, 11, c->dom_exchange, 0);
    if (!rc) rc = dom_grant(c, startup, DOM_LAUNCH_BYTES, 12, startup, 0);
    if (!rc) {
        u64 inherited[2] = {0, 0};
        unsigned long irqflags;
        unsigned sig;
        spin_lock_irqsave(&current->sighand->siglock, irqflags);
        inherited[0] = current->blocked.sig[0];
        for (sig = 1; sig <= 64; ++sig)
            if (current->sighand->action[sig - 1].sa.sa_handler == SIG_IGN)
                inherited[1] |= 1ULL << (sig - 1);
        spin_unlock_irqrestore(&current->sighand->siglock, irqflags);
        rc = dom_store(c, c->dom_meta + CAPSTONE_SIGNAL_OFFSET +
                        offsetof(struct capstone_signal_block, initial_mask),
                        inherited, sizeof(inherited));
    }
    mutex_unlock(&vm_lock);
    if (rc) goto out;
    WRITE_ONCE(dom_last_base, base);
    /* Plan v5 step 3 adds the trap entry and step 4 binds the context to the mm and enters
     * the program at base + entry. Until then nothing may return to the image: an ordinary
     * return would run it as an unprotected program. */
    pr_info("capstone_task: image ready at 0x%lx, entry 0x%lx: no protected entry yet, refused\n",
            base, base + (unsigned long)im.entry);
    rc = -ENOEXEC;
out:
    if (c) dom_context_destroy(c);
    kvfree(block); kfree(tail);
    /* Linux 6.1's ordinary post-PONR SIGSEGV is ptrace-suppressible. Never let a failure
     * after the point of no return recover scalar execution. */
    if (rc && bprm->point_of_no_return)
        force_exit_sig(SIGSEGV);
    return rc;
}
static struct linux_binfmt capstone_dom = {
    .module = THIS_MODULE, .load_binary = dom_load_binary,
    .capstone_native = true,
};
