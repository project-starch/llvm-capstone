#include "capstone_sqlite_vfs.h"

static void fail_fast(void) {
  __builtin_trap();
}

void sqlite_vfs_skeleton_domain_impl(unsigned *res, unsigned func,
                                     struct capstone_sqlite_vfs_bundle *bundle) {
  unsigned long sqlite_vfs_marker;

  (void)func;

  capstone_sqlite_vfs_bootstrap(bundle);
  if (bundle->io.iVersion != 1)
    fail_fast();
  if (bundle->vfs.iVersion != 3)
    fail_fast();
  if (bundle->vfs.pAppData != (void *)bundle)
    fail_fast();
  if (bundle->vfs.mxPathname != CAPSTONE_SQLITE_VFS_MAX_PATHNAME)
    fail_fast();
  if (bundle->vfs.szOsFile != (int)sizeof(struct capstone_sqlite_file))
    fail_fast();
  if (!bundle->io.xClose || !bundle->io.xRead || !bundle->io.xWrite)
    fail_fast();
  if (!bundle->vfs.xOpen || !bundle->vfs.xAccess || !bundle->vfs.xSleep)
    fail_fast();

  sqlite_vfs_marker = ((unsigned long)bundle->vfs.iVersion << 16) |
                      (unsigned long)bundle->io.iVersion;
  *res = (unsigned)sqlite_vfs_marker;
}







