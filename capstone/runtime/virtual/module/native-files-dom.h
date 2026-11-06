/* SPDX-License-Identifier: GPL-2.0-only */
/* Native fd services: authority is checked by native_acquire before issue. */
#include <linux/fcntl.h>
extern long capstone_native_fcntl(unsigned int, struct file *, unsigned int,
                                  unsigned long, struct flock *, unsigned int);
extern long capstone_native_ftruncate(struct file *, loff_t, int);
extern int capstone_native_fmetadata(struct file *, unsigned int, unsigned int, unsigned int);
struct native_wire_flock {
    s16 type, whence;
    u32 pad;
    s64 start, length;
    s32 pid;
    u32 pad2;
};
static bool native_files_issue(struct context *c, const struct ns_invocation *call,
                               struct file *file, long *result)
{
    const u64 *a = call->scalar;
    u64 nr = call->descriptor->nr;
    typeof(&capstone_native_fcntl) fcntlfn;
    typeof(&capstone_native_ftruncate) truncatefn;
    typeof(&capstone_native_fmetadata) metadatafn;
    struct native_wire_flock wire;
    struct flock lock = {0};
    long rc = -ENOSYS;
    bool duplicate;
    BUILD_BUG_ON(sizeof(wire) != 32);
    BUILD_BUG_ON(F_GETLK != NS_F_GETLK || F_SETLK != NS_F_SETLK || F_SETLKW != NS_F_SETLKW);
    switch (nr) {
    case CAPSTONE_NR_FCNTL_LOCK:
    case CAPSTONE_SYS_fcntl:
        /* An unsupported 64-bit command must not alias an accepted low word. */
        if (a[1] > UINT_MAX) { rc = -EINVAL; break; }
        if (nr == CAPSTONE_NR_FCNTL_LOCK) {
            if (a[1] != F_GETLK && a[1] != F_SETLK && a[1] != F_SETLKW) { rc = -EINVAL; break; }
            if (!call->buffer[2].data || call->buffer[2].length != sizeof(wire)) { rc = -EFAULT; break; }
            memcpy(&wire, call->buffer[2].data, sizeof(wire));
            lock.l_type = wire.type; lock.l_whence = wire.whence;
            lock.l_start = wire.start; lock.l_len = wire.length; lock.l_pid = wire.pid;
        } else if (a[1] == F_GETLK || a[1] == F_SETLK || a[1] == F_SETLKW) {
            rc = -EINVAL; break; /* pointer commands require the checked wire service */
        }
        duplicate = a[1] == F_DUPFD || a[1] == F_DUPFD_CLOEXEC;
        fcntlfn = symbol_get(capstone_native_fcntl);
        if (!fcntlfn) break;
        mutex_unlock(&vm_lock);
        rc = fcntlfn(a[0], file, a[1], a[2], nr == CAPSTONE_NR_FCNTL_LOCK ? &lock : NULL, DOM_FDS);
        mutex_lock(&vm_lock);
        symbol_put(capstone_native_fcntl);
        /* The helper reserves below DOM_FDS; no close-on-failure rollback that
         * would accidentally release the caller's existing POSIX locks. */
        if (rc >= 0 && duplicate) set_bit(rc, c->dom_delegate->fds);
        if (!rc && nr == CAPSTONE_NR_FCNTL_LOCK && a[1] == F_GETLK) {
            memset(&wire, 0, sizeof(wire));
            wire.type = lock.l_type; wire.whence = lock.l_whence;
            wire.start = lock.l_start; wire.length = lock.l_len; wire.pid = lock.l_pid;
            memcpy(call->buffer[2].data, &wire, sizeof(wire));
        }
        break;
    case CAPSTONE_SYS_fsync: case CAPSTONE_SYS_fdatasync:
        mutex_unlock(&vm_lock);
        rc = vfs_fsync(file, nr == CAPSTONE_SYS_fdatasync);
        mutex_lock(&vm_lock);
        break;
    case CAPSTONE_SYS_ftruncate:
        truncatefn = symbol_get(capstone_native_ftruncate);
        if (!truncatefn) break;
        mutex_unlock(&vm_lock);
        rc = truncatefn(file, (s64)a[1], 0);
        mutex_lock(&vm_lock);
        symbol_put(capstone_native_ftruncate);
        break;
    case CAPSTONE_SYS_fchmod: case CAPSTONE_SYS_fchown:
        metadatafn = symbol_get(capstone_native_fmetadata);
        if (!metadatafn) break;
        mutex_unlock(&vm_lock);
        rc = metadatafn(file, nr == CAPSTONE_SYS_fchmod ? 0 : 1, a[1], a[2]);
        mutex_lock(&vm_lock);
        symbol_put(capstone_native_fmetadata);
        break;
    default: return false;
    }
    *result = rc;
    return true;
}
