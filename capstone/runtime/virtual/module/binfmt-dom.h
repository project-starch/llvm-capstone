/* SPDX-License-Identifier: GPL-2.0-only */
/* Experimental native exec driver. Included after the ordinary device ops so
 * the existing arena/context lifecycle stays the single implementation. */
#include <linux/binfmts.h>
#include <linux/elf.h>
#include <linux/anon_inodes.h>
#include <linux/mman.h>
#include <linux/uio.h>
#include <linux/shmem_fs.h>
#include <linux/fs_struct.h>
#include <linux/path.h>
#include <linux/cred.h>
#include <linux/pid_namespace.h>
#include <linux/ptrace.h>
#include <linux/security.h>
#include <asm/processor.h>
#include <asm/cacheflush.h>
#include "capstone/delegate.h"

/* Read-only attribution records; sampled outside measurement windows. */
static unsigned long dom_last_base, dom_last_trampoline;
module_param(dom_last_base, ulong, 0400);
module_param(dom_last_trampoline, ulong, 0400);
static bool dom_binfmt;
module_param(dom_binfmt, bool, 0444);
MODULE_PARM_DESC(dom_binfmt, "experimental root-only native .dom exec (default off)");
#define DOM_STEP _IO('V', 31)
#define CV_STEP_NATIVE _IO('V', 34)
#define DOM_PROTECT_DONE _IO('V', 32)
#define DOM_DO_PROTECT 257
#define DOM_DO_SYSCALL 258
#define DOM_SYSCALL_DONE _IO('V', 33)
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
                     unsigned reg, unsigned perms, unsigned long cursor, bool linear,
                     unsigned long rx)
{
    struct cv_map m = {.address = base, .bytes = bytes, .reg = reg,
                      .permissions = perms, .cursor = cursor, .linear = linear, .rx_bytes = rx};
#ifdef CV_RACE_TEST
    /* The racer's MAP service grants into the racer's own frame. */
    if (rt.map_for_b && rt.b) return add(c, rt.b, &m);
#endif
    return add(c, c->main_thread, &m);
}
static long dom_map(struct context *c, struct cv_step *s)
{
    unsigned long bytes = s->args[0], align = s->args[1], base;
    unsigned prot = s->args[2], rights = s->args[3], kind = s->args[4];
    int rc;
    if (!bytes || bytes > CV_MAX_REGION_BYTES || (bytes & ~PAGE_MASK) ||
        align < PAGE_SIZE || align > CV_MAX_REGION_BYTES || !is_power_of_2(align) ||
        prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC) || kind > CV_MAP_METADATA ||
        (kind != CV_MAP_APPLICATION && (prot != (PROT_READ | PROT_WRITE) || rights != 6)) ||
        (kind == CV_MAP_APPLICATION && rights != 7)) return -EINVAL;
    align = max(align, capstone_repr_granule(bytes));
    if (bytes & (align - 1)) return -EINVAL;
    base = dom_reserve(bytes, align, prot);
    if (IS_ERR_VALUE(base)) return base;
    rc = dom_grant(c, base, bytes, 10, rights, base, true, 0);
    if (rc) vm_munmap(base, bytes);
    else {
        struct arena *a = arena_at(c, base);
        a->dom_dynamic = true;
        a->dom_metadata = kind == CV_MAP_METADATA;
    }
    return rc;
}
static long dom_unmap(struct context *c, unsigned long base, unsigned long bytes)
{
    struct arena *a = find_arena(c, base, bytes);
    int rc;
    bool held;
    if (!a || !a->dom_dynamic || a->dom_metadata) return -EINVAL;
    held = a->copy_holds != 0;
    rc = retire(c, a, false);
    if (!rc && held) {
        /* retire retained the held arena; keep its VMA as a VA reservation.
         * Last release performs physical unmap after all checked windows. */
        a->copy_unmap_pending = true;
        return 0;
    }
    return rc ? rc : vm_munmap(base, bytes);
}

#ifdef CV_RACE_TEST
#include "rt-dom.h"
#endif
#include "delegate-dom.h"
#include "native-syscall-dom.h"

/* The transport is fixed at exec, unretirable and only this task runs the
 * domain. Thus no in-flight service can race its reuse while vm_lock is down.
 * Copies into the caller's object remain capability accesses in the SDK. */
static long dom_rw(struct context *c, const struct capstone_delegate_entry *e)
{
    struct file *file;
    struct iovec iov;
    struct iov_iter iter;
    loff_t pos = 0, *ppos = NULL;
    bool write = e->nr == 64, locked = false;
    unsigned long buf;
    long rc;
    if (e->count != 1 || e->flags != 2 ||
        !dom_within(e->args[1], e->args[2], c->dom_exchange_bytes)) return -EFAULT;
    if (e->args[0] > INT_MAX) return -EBADF;
    if (!dom_fd_allowed(c, e->args[0])) return -EBADF;
    file = fget(e->args[0]);
    if (!file) return -EBADF;
    /* The private context descriptor is never an application I/O target. */
    if (file->private_data == c || !file->f_op) { rc = -EBADF; goto put; }
    if (!(write ? file->f_op->write_iter : file->f_op->read_iter)) {
        /* Console descriptors can have only the legacy .read/.write API.
         * Use the existing private bounce path and Linux syscall checks;
         * never invoke a file operation with a kernel or domain pointer. */
        fput(file);
        return dom_prepare(c, e);
    }
    buf = c->dom_exchange + e->args[1];
    if (!write && (rc = dom_touch(c, buf, e->args[2]))) goto put;
    rc = import_single_range(write ? WRITE : READ, (void __user *)buf, e->args[2], &iov, &iter);
    if (rc) goto put;
    mutex_unlock(&vm_lock);
    if (file->f_mode & FMODE_ATOMIC_POS) { mutex_lock(&file->f_pos_lock); locked = true; }
    if (!(file->f_mode & FMODE_STREAM)) { pos = file->f_pos; ppos = &pos; }
    if (write) {
        file_start_write(file);
        rc = vfs_iter_write(file, &iter, ppos, 0);
        file_end_write(file);
    } else rc = vfs_iter_read(file, &iter, ppos, 0);
    if (ppos && rc >= 0) file->f_pos = pos;
    if (locked) mutex_unlock(&file->f_pos_lock);
    mutex_lock(&vm_lock);
    if (rc <= -ERESTARTSYS && rc >= -ERESTART_RESTARTBLOCK) rc = -EINTR;
put:
    fput(file);
    return rc;
}
static long dom_protect(struct context *c, struct cv_step *s)
{
    unsigned long base = s->args[0], bytes = s->args[1], prot = s->args[2];
    struct arena *a = arena_at(c, base);
    struct pt_regs *regs = current_pt_regs();
    if (!bytes || ((base | bytes) & ~PAGE_MASK) ||
        (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC)) || !a || !a->dom_dynamic ||
        a->copy_holds || a->copy_retired ||
        base < a->address || !dom_within(base - a->address, bytes, a->bytes)) {
        s->result = -EINVAL;
        return 0;
    }
    regs->a1 = base; regs->a2 = bytes; regs->a3 = prot;
    c->dom_protect_pending = true;
    return DOM_DO_PROTECT;
}

static int dom_reply(struct context *c, long result)
{
    u64 words[3] = {result, 0, CAPSTONE_ROUND_DONE};
    u32 zero = 0;
    int rc = dom_store(c, c->dom_meta + offsetof(struct capstone_delegate_entry, result),
                       words, sizeof(words));
    if (!rc) rc = dom_store(c, c->dom_meta + CAPSTONE_SIGNAL_OFFSET +
                            offsetof(struct capstone_signal_block, count), &zero, sizeof(zero));
    c->dom_step.result = 0;
    return rc;
}

static long dom_service(struct context *c, struct cv_step *s)
{
    struct capstone_delegate_entry e;
    size_t lengths[CAPSTONE_DELEGATE_ARGS];
    long rc;
    s->reply = 1;
    s->result = -ENOSYS;
    if (native_dispatch && (s->args[7] == NS_SERVICE_NATIVE ||
        s->args[7] == CV_SERVICE_DELEGATE || s->args[7] == CV_SERVICE_THREAD_DELEGATE))
        return native_service(c, s);
#ifdef CV_RACE_TEST
    if (rt_service(c, s)) return 0;
#endif
    switch (s->args[7]) {
    case CV_SERVICE_THREAD_SELF: s->result = 1; break;
    case CV_SERVICE_PROTECT: if (native_dispatch) {
            rc = native_protect(c, s);
            if (rc) s->result = rc;
            return 0;
        }
        return dom_protect(c, s);
    case CV_SERVICE_MAP: s->result = dom_map(c, s); break;
    case CV_SERVICE_UNMAP: s->result = dom_unmap(c, s->args[0], s->args[1]); break;
    case CV_SERVICE_NODES:
        s->result = s->args[0] > CV_ID_RECORDS_MAX - CV_NODE_RESERVE ? -ENOMEM :
                    ensure_nodes(c, s->args[0] + CV_NODE_RESERVE);
        break;
    case CV_SERVICE_WAIT: s->result = 0; break;
    case CV_SERVICE_DELEGATE:
    case CV_SERVICE_THREAD_DELEGATE:
        if (s->args[7] == CV_SERVICE_THREAD_DELEGATE &&
            (s->args[0] != c->dom_meta || s->args[1] != c->dom_exchange ||
             s->args[2] != c->dom_exchange_bytes)) return -EFAULT;
        if (copy_from_user(&e, (void __user *)c->dom_meta, sizeof(e))) return -EFAULT;
        rc = capstone_delegate_validate(&e, NULL, c->dom_exchange_bytes, lengths);
        if (rc) return dom_reply(c, -rc);
        if (e.nr == CAPSTONE_SYS_exit_group) return (e.args[0] & 255) + 1;
        if (e.nr == DOM_HELLO || e.nr == CAPSTONE_NR_SIGPOLL) e.result = 0;
        else {
            rc = (e.nr == CAPSTONE_SYS_read || e.nr == CAPSTONE_SYS_write) ?
                 dom_rw(c, &e) : dom_prepare(c, &e);
            /* A brokered open may legitimately return fd 258. Only an
             * actual prepared request authorizes scalar syscall dispatch. */
            if (rc == DOM_DO_SYSCALL && c->dom_delegate->pending) return rc;
            e.result = rc;
        }
        rc = dom_reply(c, e.result);
        if (rc) return rc;
        break;
    }
    return 0;
}
static long dom_ioctl(struct file *f, unsigned int op, unsigned long arg)
{
    struct context *c = f->private_data;
    struct cv_step *s = &c->dom_step;
    long rc;
    u64 deadline = ktime_get_ns() + CV_STEP_BUDGET_NS;
    if (current->mm != c->mm || task_pid_nr(current) != c->dom_pid) return -EPERM;
    if (op == DOM_PROTECT_DONE) {
        /* Only the scalar trampoline can issue this operation. Complete even
         * with a pending signal: otherwise an interrupted ioctl loses result. */
        if (!c->dom_protect_pending || (long)arg > 0 || (long)arg < -4095) return -EINVAL;
        c->dom_protect_pending = false;
        s->result = (long)arg;
        return 0;
    }
    if (op == DOM_SYSCALL_DONE) {
        long result = (long)arg;
        /* Finish under a noninterruptible lock. A signal must not lose the
         * result of an already executed syscall or run it twice. */
        if (!c->dom_delegate || !c->dom_delegate->pending) return -EINVAL;
        mutex_lock(&vm_lock);
        rc = dom_complete(c, &result);
        if (!rc) rc = dom_reply(c, result);
        c->dom_delegate->pending = false;
        if (rc) goto fault;
        goto out;
    }
    if (op != (native_dispatch ? CV_STEP_NATIVE : DOM_STEP) || arg || c->dom_protect_pending ||
        (c->dom_delegate && c->dom_delegate->pending)) return -EINVAL;
    if (mutex_lock_interruptible(&vm_lock)) return -EINTR;
native_again:
    rc = cv_step_locked(c, s);
    if (rc == -EINTR || rc == -EAGAIN) goto out;
    if (rc) goto fault;
    s->reply = 0;
    if (s->kind == 1) goto out;
    if (s->kind == 4) {
        if (native_dispatch && cv_step_stop(deadline)) goto out;
        rc = resolve(c, c->main_thread, ktime_get_ns() + CV_STEP_BUDGET_NS);
        if (!rc) {
            c->main_thread->sfence_pending = true;
            if (native_dispatch && !cv_step_stop(deadline)) goto native_again;
            goto out;
        }
        goto fault;
    }
    if (s->kind != 3) { rc = -EFAULT; goto fault; }
    if (native_dispatch) TEST_HOOK(5);
    rc = dom_service(c, s);
    if (!rc && native_dispatch && !cv_step_stop(deadline)) goto native_again;
    if (rc >= 0) goto out;
fault:
    pr_info("capstone_dom: fault rc=%ld kind=%llu cause=%llu pc=0x%llx address=0x%llx\n",
            rc, s->kind, s->cause, s->pc, s->address);
    send_sig(SIGSEGV, current, 1);
    rc = 128 + SIGSEGV + 1;
out:
    TEST_IOCTL_DONE();
    slot_changed();
    mutex_unlock(&vm_lock);
    return rc;
}
static int dom_release(struct inode *inode, struct file *f)
{
    if (!f->private_data) return 0;
    {
        struct context *c = f->private_data;
#ifdef CV_RACE_TEST
        /* The racer thread dies with its context. */
        if (rt.b && rt.b->context == c) { rt.b = NULL; rt.armed = false; rt.c = NULL; }
#endif
        ns_session_free(c->native_session);
        kfree(c->native_context);
    }
    dom_delegate_free(f->private_data);
    return vm_release(inode, f);
}
static const struct file_operations dom_ops = {
    .owner = THIS_MODULE, .release = dom_release, .unlocked_ioctl = dom_ioctl,
};

static int dom_startup(struct linux_binprm *bprm, void *block)
{
    struct dom_launch_header *h = block;
    u32 *offsets = block + sizeof(*h);
    unsigned long at = bprm->p, used = sizeof(*h) + (bprm->argc + bprm->envc) * sizeof(u32);
    struct path pwd;
    char *cwd, *tmp;
    unsigned i;
    memcpy(h->magic, "CPLAUNCH", 8);
    h->version = 2; h->argc = bprm->argc; h->envc = bprm->envc;
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

/* Pure RV64I, no libc, stack access or capability authority; s0 is the descriptor and s1 the ioctl number from load_binary. */
static const u32 dom_trampoline[] = {
    0x00040513, 0x00048593, 0x00000613, 0x01d00893, 0x00000073,
    0x10100293, 0x02550063, 0x10200293, 0x04550263, 0xfca05ee3,
    0xfff50513, 0x05e00893, 0x00000073, 0x0000006f, 0x00058513,
    0x00060593, 0x00068613, 0x0e200893, 0x00000073, 0x00050613,
    0x00040513, 0x00090593, 0x01d00893, 0x00000073, 0xfa1ff06f,
    0x00058513, 0x00060593, 0x00068613, 0x00070693, 0x00078713,
    0x00080793, 0x00000073, 0x00050613, 0x00040513, 0x00098593,
    0x01d00893, 0x00000073, 0xf6dff06f,
};
static struct linux_binfmt capstone_dom;
static int dom_load_binary(struct linux_binprm *bprm)
{
    struct dom_image im = {0};
    struct file *ctxfile = NULL, *trampfile = NULL;
    struct context *c;
    struct pt_regs *regs = current_pt_regs();
    unsigned long base, stack, startup, tramp, whole, part;
    void *block = NULL, *tail = NULL;
    int rc, fd = -1;
    loff_t pos = 0;
    Elf64_Ehdr *h = (void *)bprm->buf;
    if (memcmp(h->e_ident, ELFMAG, SELFMAG) || h->e_machine != 259) return -ENOEXEC;
    if (!uid_eq(current_euid(), GLOBAL_ROOT_UID) || bprm->secureexec ||
        (bprm->unsafe & LSM_UNSAFE_SHARE) ||
        (file_inode(bprm->file)->i_mode & (S_ISUID | S_ISGID))) return -EACCES;
    if (!one_inode_per_file(bprm->file) || !file_inode(bprm->file)->i_nlink) return -EOPNOTSUPP;
    if (bprm->argc < 1 || bprm->argc > DOM_MAX_STRINGS ||
        bprm->envc > DOM_MAX_STRINGS - bprm->argc) return -E2BIG;
    if (native_dispatch && (!checked_copy || !native_slot_transport ||
        current->seccomp.mode != SECCOMP_MODE_DISABLED)) return -EOPNOTSUPP;
    rc = dom_parse(bprm->file, &im);
    if (rc) return rc;
    block = kvzalloc(DOM_LAUNCH_BYTES, GFP_KERNEL);
    tail = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!block || !tail) { rc = -ENOMEM; goto out; }
    /* Read the partial data page before the point of no return. */
    whole = im.data.p_filesz & PAGE_MASK; part = im.data.p_filesz - whole;
    if (part && (rc = dom_read(bprm->file, tail, part, im.data.p_offset + whole))) goto out;
    trampfile = shmem_file_setup("capstone-trampoline", PAGE_SIZE, VM_NORESERVE);
    if (IS_ERR(trampfile)) { rc = PTR_ERR(trampfile); trampfile = NULL; goto out; }
    rc = kernel_write(trampfile, dom_trampoline, sizeof(dom_trampoline), &pos);
    if (rc != sizeof(dom_trampoline)) { rc = rc < 0 ? rc : -EIO; goto out; }
    rc = begin_new_exec(bprm);
    if (rc) goto out;
    setup_new_exec(bprm);
    rc = setup_arg_pages(bprm, STACK_TOP, EXSTACK_DISABLE_X);
    if (rc) goto out;
    rc = dom_startup(bprm, block);
    if (rc) goto out;
    ctxfile = anon_inode_getfile("capstone-dom", &dom_ops, NULL, O_RDWR);
    if (IS_ERR(ctxfile)) { rc = PTR_ERR(ctxfile); ctxfile = NULL; goto out; }
    rc = vm_open(NULL, ctxfile);
    if (rc) goto out;
    c = ctxfile->private_data;
    c->dom_fd = -1;
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
    rc = dom_grant(c, base, im.bytes, 3, 7, base + im.entry, false, im.text.p_memsz);
    if (!rc) rc = dom_grant(c, stack, im.stack, 2, 6, stack + im.stack, false, 0);
    if (!rc) rc = dom_grant(c, c->dom_meta, DOM_META_BYTES, 10, 6, c->dom_meta, false, 0);
    if (!rc) rc = dom_grant(c, c->dom_exchange, im.exchange, 11, 6, c->dom_exchange, false, 0);
    if (!rc) rc = dom_grant(c, startup, DOM_LAUNCH_BYTES, 12, 4, startup, false, 0);
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
    tramp = vm_mmap(trampfile, 0, PAGE_SIZE, PROT_READ | PROT_EXEC, MAP_PRIVATE, 0);
    if (IS_ERR_VALUE(tramp)) { rc = tramp; goto out; }
    fd = get_unused_fd_flags(O_CLOEXEC);
    if (fd < 0) { rc = fd; goto out; }
    c->dom_pid = task_pid_nr(current);
    c->dom_fd = fd;
    WRITE_ONCE(dom_last_base, base);
    WRITE_ONCE(dom_last_trampoline, tramp);
    current->mm->start_code = base;
    current->mm->end_code = base + im.text.p_memsz;
    current->mm->start_data = base + im.text.p_memsz;
    current->mm->end_data = base + im.text.p_memsz + im.data.p_memsz;
    current->mm->start_stack = bprm->p;
    set_binfmt(&capstone_dom);
    finalize_exec(bprm);
    /* start_thread is not exported on this kernel. The scalar trampoline
     * uses no FP; begin_new_exec already reset thread/FPU state. */
    memset(regs, 0, sizeof(*regs));
    regs->status = SR_PIE | SR_UXL_64;
    regs->epc = tramp; regs->sp = (bprm->p - 16) & ~15UL;
    regs->s0 = fd; regs->s1 = native_dispatch ? CV_STEP_NATIVE : DOM_STEP; regs->s2 = DOM_PROTECT_DONE; regs->s3 = DOM_SYSCALL_DONE;
    fd_install(fd, ctxfile); ctxfile = NULL;
    rc = 0;
out:
    if (ctxfile) fput(ctxfile);
    if (trampfile) fput(trampfile);
    kvfree(block); kfree(tail);
    return rc;
}
static struct linux_binfmt capstone_dom = {
    .module = THIS_MODULE, .load_binary = dom_load_binary,
};
