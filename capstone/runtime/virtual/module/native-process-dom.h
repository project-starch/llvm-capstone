/* SPDX-License-Identifier: GPL-2.0-only */
/* pipe2 is a resource transaction: reserve, checked publication, then install.
 * No descriptor is installed if either the payload or exchange reply fails.
 * The protected task cannot resume until the infallible install has finished. */
static int native_exchange_reply(struct context *, const struct cn_capref *,
                                 const struct capstone_delegate_entry *, long);

static void native_pipe_abort(int fd[2], struct file *files[2])
{
    fput(files[0]); fput(files[1]);
    put_unused_fd(fd[0]); put_unused_fd(fd[1]);
}
static void native_pipe_commit(struct context *c, int fd[2], struct file *files[2])
{
    set_bit(fd[0], c->dom_delegate->fds);
    set_bit(fd[1], c->dom_delegate->fds);
    fd_install(fd[0], files[0]); fd_install(fd[1], files[1]);
}
static long native_pipe_stage(struct context *c, const struct cn_capref *out,
                              u64 offset, u64 flags, int fd[2], struct file *files[2])
{
    struct cu_result copied;
    long rc;
    if (current->seccomp.mode != SECCOMP_MODE_DISABLED) return -EOPNOTSUPP;
    if (flags & ~(u64)(O_CLOEXEC | O_NONBLOCK)) return -EINVAL;
    rc = cn_authorize(c->native_context, &c->native_context->owner,
                       out, offset, 2 * sizeof(int), 1);
    if (rc) return rc;
    mutex_unlock(&vm_lock);
    rc = capstone_native_pipe_prepare(fd, files, flags, DOM_FDS);
    mutex_lock(&vm_lock);
    if (rc) return rc;
    copied = cv_checked_copy(c, c->main_thread, out, offset, fd,
                             2 * sizeof(int), CU_TO_DOMAIN);
    if (copied.cause || copied.bytes_done != 2 * sizeof(int)) {
        native_pipe_abort(fd, files);
        return -EFAULT;
    }
    return 0;
}
static long native_pipe_direct(struct context *c, struct cv_step *step)
{
    struct cv_thread *t = c->main_thread;
    struct cn_capref out = {t->id, READ_ONCE(t->frame[CV_FRAME_CHECK_EPOCH]), 10, 0};
    struct file *files[2];
    int fd[2];
    long rc = ns_session_admit(c->native_session, out.context_epoch);
    if (!rc && !nf_regs_plain(t, READ_ONCE(t->frame[CV_FRAME_CHECK_KINDS]), 1ULL << 10)) rc = -EINVAL;
    if (!rc) {
        rc = native_pipe_stage(c, &out, 0, step->args[1], fd, files);
        if (!rc) native_pipe_commit(c, fd, files);
    }
    step->result = rc;
    return 0;
}
static long native_pipe_exchange(struct context *c, const struct cn_capref *meta,
                                 const struct cn_capref *exchange,
                                 const struct capstone_delegate_entry *e)
{
    struct file *files[2];
    int fd[2];
    long rc = ns_session_admit(c->native_session, meta->context_epoch), reply;
    if (!rc) rc = native_pipe_stage(c, exchange, e->args[0], e->args[1], fd, files);
    reply = native_exchange_reply(c, meta, e, rc);
    if (!rc) {
        if (reply) native_pipe_abort(fd, files);
        else native_pipe_commit(c, fd, files);
    }
    return reply;
}
