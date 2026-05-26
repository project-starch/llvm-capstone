#ifndef CAPSTONE_SQLITE_VFS_H
#define CAPSTONE_SQLITE_VFS_H

#include "sqlite3.h"

#define CAPSTONE_SQLITE_VFS_NAME "capstone-hostcall"
#define CAPSTONE_SQLITE_VFS_MAX_PATHNAME 256
#define CAPSTONE_SQLITE_VFS_SECTOR_SIZE 4096

typedef unsigned long capstone_hostcall_handle_t;

enum capstone_sqlite_vfs_check_result {
  CAPSTONE_SQLITE_VFS_CHECK_OK = 0,
  CAPSTONE_SQLITE_VFS_CHECK_NULL = 1,
  CAPSTONE_SQLITE_VFS_CHECK_IO_VERSION = 2,
  CAPSTONE_SQLITE_VFS_CHECK_VFS_VERSION = 3,
  CAPSTONE_SQLITE_VFS_CHECK_FILE_SIZE = 4,
  CAPSTONE_SQLITE_VFS_CHECK_APPDATA = 5,
  CAPSTONE_SQLITE_VFS_CHECK_NAME = 6,
  CAPSTONE_SQLITE_VFS_CHECK_IO_POINTERS = 7,
  CAPSTONE_SQLITE_VFS_CHECK_VFS_POINTERS = 8,
};

struct capstone_sqlite_file {
  sqlite3_file base;
  capstone_hostcall_handle_t handle;
  int current_lock;
  int open_flags;
};

struct capstone_sqlite_vfs_bundle {
  sqlite3_io_methods io;
  sqlite3_vfs vfs;
};

void capstone_sqlite_vfs_bootstrap(struct capstone_sqlite_vfs_bundle *bundle);
int capstone_sqlite_vfs_self_check(const struct capstone_sqlite_vfs_bundle *bundle);

#endif

