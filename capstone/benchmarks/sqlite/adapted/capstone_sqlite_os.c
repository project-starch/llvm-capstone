#include "capstone_sqlite_vfs.h"

static struct capstone_sqlite_vfs_bundle capstone_sqlite_bundle;

int sqlite3_os_init(void) {
  capstone_sqlite_vfs_bootstrap(&capstone_sqlite_bundle);
  return sqlite3_vfs_register(&capstone_sqlite_bundle.vfs, 1);
}

int sqlite3_os_end(void) { return SQLITE_OK; }
