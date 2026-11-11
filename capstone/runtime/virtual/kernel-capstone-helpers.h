/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef _LINUX_CAPSTONE_NATIVE_HELPERS_H
#define _LINUX_CAPSTONE_NATIVE_HELPERS_H

#include <linux/linkage.h>
#include <linux/types.h>
#include <linux/capstone_native_spawn.h>
#include <linux/capstone_native_time.h>

struct dentry;
struct file;
struct flock;
struct kstat;
struct path;
struct rlimit;
struct stat;
struct sysinfo;

/* Internal built-in adapter interface, not a module or userspace ABI.
 * The adapter authenticates capabilities and owns resource admission. Helpers
 * receive held Linux objects and kernel buffers, with the caller's existing
 * locking and lifetime rules. Missing implementations are link-time errors. */
int capstone_unpinned_config(unsigned int ids, bool *exact);
int capstone_native_mprotect(unsigned long start, size_t len, unsigned long prot);
int capstone_native_sysinfo(struct sysinfo *info);
int capstone_native_getrlimit(unsigned int resource, struct rlimit *limit);
int capstone_native_stat(struct kstat *stat, struct stat *statbuf);

long capstone_native_fcntl(unsigned int fd, struct file *file, unsigned int cmd,
                          unsigned long arg, struct flock *lock, unsigned int limit);
long capstone_native_ftruncate(struct file *file, loff_t length, int small);
int capstone_native_fmetadata(struct file *file, unsigned int op,
                             unsigned int first, unsigned int second);
int capstone_native_reserve_fd(unsigned int flags, unsigned int limit);
int capstone_native_dupfd(unsigned long from, struct file *file,
                         unsigned int flags, unsigned int limit);
int capstone_native_pipe_prepare(int fd[2], struct file *files[2],
                                unsigned int flags, unsigned int limit);
int capstone_native_signal_action(unsigned int sig, unsigned int cls,
                                 unsigned int flags);
int capstone_native_path_chmod(const struct path *path, umode_t mode);
int capstone_native_path_link(const struct path *source, const struct path *parent,
                             struct dentry *new_dentry);

#endif
