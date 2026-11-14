/* SPDX-License-Identifier: GPL-2.0-only */
/* The task-owned resource capsule: the directories, devices and descriptors a protected
 * program may use, and its spawned children. */
#include "delegate-wire.c"
#include <linux/namei.h>
#include <linux/mount.h>
#include <linux/magic.h>
#include <linux/kdev_t.h>
#include <linux/fdtable.h>

#define DOM_FDS 1024
#define DOM_EXTRA_DIRS 2
#define DOM_DEVICES 3
#define DOM_CHILDREN 64

struct dom_delegate {
    struct path root, cwd_path; /* immutable grant; mutable Linux working directory */
    struct path dirs[DOM_EXTRA_DIRS], devices[DOM_DEVICES];
    char *root_name; /* admission name only; live names come from pinned paths */
    DECLARE_BITMAP(fds, DOM_FDS);
    struct file *shell; /* explicitly provisioned executable, no directory authority */
    struct pid *children[DOM_CHILDREN];
};

static int dom_spawn_resources(struct context *c);
#include "resource-dom.h"

/* A plain store into the program's memory, before the program runs. The page-tag code keeps
 * tags with physical pages, so no page needs to be held for it. */
static int dom_store(struct context *c, unsigned long at, const void *src, size_t bytes)
{
    (void)c;
    return copy_to_user((void __user *)at, src, bytes) ? -EFAULT : 0;
}
static void dom_delegate_free(struct context *c)
{
    struct dom_delegate *d = c->dom_delegate;
    if (!d) return;
    path_put(&d->root);
    path_put(&d->cwd_path);
    for (unsigned i = 0; i < DOM_EXTRA_DIRS; ++i)
        if (d->dirs[i].dentry) path_put(&d->dirs[i]);
    for (unsigned i = 0; i < DOM_DEVICES; ++i)
        if (d->devices[i].dentry) path_put(&d->devices[i]);
    if (d->shell) fput(d->shell);
    for (unsigned i = 0; i < DOM_CHILDREN; ++i)
        if (d->children[i]) put_pid(d->children[i]);
    kfree(d->root_name);
    kfree(d); c->dom_delegate = NULL;
}
