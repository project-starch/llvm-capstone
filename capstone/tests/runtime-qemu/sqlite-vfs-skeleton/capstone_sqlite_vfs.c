#include "capstone_sqlite_vfs.h"

/*
 * Keep this skeleton source free of byte-wise capability-pointer loops for now.
 * The current Capstone backend still miscompiles or crashes on a few optimized
 * forms of that pattern, and this smoke only needs bootstrap/self-check paths.
 */

static struct capstone_sqlite_file *capstone_sqlite_file_from_base(sqlite3_file *file) {
  return (struct capstone_sqlite_file *)file;
}

static struct capstone_sqlite_vfs_bundle *capstone_sqlite_bundle_from_vfs(sqlite3_vfs *vfs) {
  return (struct capstone_sqlite_vfs_bundle *)vfs->pAppData;
}

static void capstone_sqlite_file_reset(struct capstone_sqlite_file *cap_file) {
  cap_file->base.pMethods = 0;
  cap_file->handle = 0;
  cap_file->current_lock = SQLITE_LOCK_NONE;
  cap_file->open_flags = 0;
}

static int capstone_sqlite_io_close(sqlite3_file *file) {
  struct capstone_sqlite_file *cap_file = capstone_sqlite_file_from_base(file);
  capstone_sqlite_file_reset(cap_file);
  return SQLITE_OK;
}

static int capstone_sqlite_io_read(sqlite3_file *file, void *buf, int amount,
                                   sqlite3_int64 offset) {
  (void)file;
  (void)buf;
  (void)amount;
  (void)offset;
  return SQLITE_IOERR_SHORT_READ;
}

static int capstone_sqlite_io_write(sqlite3_file *file, const void *buf, int amount,
                                    sqlite3_int64 offset) {
  (void)file;
  (void)buf;
  (void)amount;
  (void)offset;
  return SQLITE_IOERR_WRITE;
}

static int capstone_sqlite_io_truncate(sqlite3_file *file, sqlite3_int64 size) {
  (void)file;
  (void)size;
  return SQLITE_IOERR_TRUNCATE;
}

static int capstone_sqlite_io_sync(sqlite3_file *file, int flags) {
  (void)file;
  (void)flags;
  return SQLITE_IOERR_FSYNC;
}

static int capstone_sqlite_io_file_size(sqlite3_file *file, sqlite3_int64 *size_out) {
  (void)file;
  if (size_out)
    *size_out = 0;
  return SQLITE_OK;
}

static int capstone_sqlite_io_lock(sqlite3_file *file, int lock_level) {
  struct capstone_sqlite_file *cap_file = capstone_sqlite_file_from_base(file);
  cap_file->current_lock = lock_level;
  return SQLITE_OK;
}

static int capstone_sqlite_io_unlock(sqlite3_file *file, int lock_level) {
  struct capstone_sqlite_file *cap_file = capstone_sqlite_file_from_base(file);
  cap_file->current_lock = lock_level;
  return SQLITE_OK;
}

static int capstone_sqlite_io_check_reserved_lock(sqlite3_file *file, int *reserved_out) {
  struct capstone_sqlite_file *cap_file = capstone_sqlite_file_from_base(file);
  if (reserved_out)
    *reserved_out = (cap_file->current_lock >= SQLITE_LOCK_RESERVED) ? 1 : 0;
  return SQLITE_OK;
}

static int capstone_sqlite_io_file_control(sqlite3_file *file, int op, void *arg) {
  (void)file;
  (void)op;
  (void)arg;
  return SQLITE_NOTFOUND;
}

static int capstone_sqlite_io_sector_size(sqlite3_file *file) {
  (void)file;
  return CAPSTONE_SQLITE_VFS_SECTOR_SIZE;
}

static int capstone_sqlite_io_device_characteristics(sqlite3_file *file) {
  (void)file;
  return 0;
}

static int capstone_sqlite_vfs_open(sqlite3_vfs *vfs, sqlite3_filename name,
                                    sqlite3_file *file, int flags,
                                    int *out_flags) {
  struct capstone_sqlite_vfs_bundle *bundle = capstone_sqlite_bundle_from_vfs(vfs);
  struct capstone_sqlite_file *cap_file = capstone_sqlite_file_from_base(file);

  (void)name;
  capstone_sqlite_file_reset(cap_file);
  cap_file->base.pMethods = &bundle->io;
  cap_file->open_flags = flags;
  if (out_flags)
    *out_flags = flags;
  return SQLITE_CANTOPEN;
}

static int capstone_sqlite_vfs_delete(sqlite3_vfs *vfs, const char *name, int sync_dir) {
  (void)vfs;
  (void)name;
  (void)sync_dir;
  return SQLITE_IOERR_DELETE;
}

static int capstone_sqlite_vfs_access(sqlite3_vfs *vfs, const char *name, int flags,
                                      int *result_out) {
  (void)vfs;
  (void)name;
  if (result_out)
    *result_out = 0;

  if (flags == SQLITE_ACCESS_EXISTS || flags == SQLITE_ACCESS_READWRITE ||
      flags == SQLITE_ACCESS_READ)
    return SQLITE_OK;

  return SQLITE_IOERR_ACCESS;
}

static int capstone_sqlite_vfs_full_pathname(sqlite3_vfs *vfs, const char *name,
                                             int out_size, char *out) {
  (void)vfs;
  if (out_size <= 0 || !out)
    return SQLITE_CANTOPEN;

  out[0] = '\0';
  if (name && name[0] != '\0') {
    if (out_size < 2)
      return SQLITE_CANTOPEN;
    out[0] = name[0];
    out[1] = '\0';
  }
  return SQLITE_OK;
}

static void *capstone_sqlite_vfs_dl_open(sqlite3_vfs *vfs, const char *filename) {
  (void)vfs;
  (void)filename;
  return 0;
}

static void capstone_sqlite_vfs_dl_error(sqlite3_vfs *vfs, int size, char *out) {
  (void)vfs;
  if (size > 0 && out)
    out[0] = '\0';
}

static void (*capstone_sqlite_vfs_dl_sym(sqlite3_vfs *vfs, void *handle,
                                         const char *symbol))(void) {
  (void)vfs;
  (void)handle;
  (void)symbol;
  return 0;
}

static void capstone_sqlite_vfs_dl_close(sqlite3_vfs *vfs, void *handle) {
  (void)vfs;
  (void)handle;
}

static int capstone_sqlite_vfs_randomness(sqlite3_vfs *vfs, int byte_count, char *out) {
  (void)vfs;
  if (byte_count <= 0 || !out)
    return 0;
  out[0] = (char)0x5a;
  return 1;
}

static int capstone_sqlite_vfs_sleep(sqlite3_vfs *vfs, int microseconds) {
  (void)vfs;
  return microseconds;
}

#ifdef SQLITE_OMIT_FLOATING_POINT
static int capstone_sqlite_vfs_current_time(sqlite3_vfs *vfs,
                                            sqlite3_int64 *out) {
#else
static int capstone_sqlite_vfs_current_time(sqlite3_vfs *vfs, double *out) {
#endif
  (void)vfs;
  if (out)
    *out = 0.0;
  return SQLITE_OK;
}

static int capstone_sqlite_vfs_get_last_error(sqlite3_vfs *vfs, int size, char *out) {
  (void)vfs;
  if (size > 0 && out)
    out[0] = '\0';
  return 0;
}

static int capstone_sqlite_vfs_current_time_int64(sqlite3_vfs *vfs,
                                                  sqlite3_int64 *out) {
  (void)vfs;
  if (out)
    *out = 0;
  return SQLITE_OK;
}

static int capstone_sqlite_vfs_set_system_call(sqlite3_vfs *vfs, const char *name,
                                               sqlite3_syscall_ptr syscall_ptr) {
  (void)vfs;
  (void)name;
  (void)syscall_ptr;
  return SQLITE_NOTFOUND;
}

static sqlite3_syscall_ptr capstone_sqlite_vfs_get_system_call(sqlite3_vfs *vfs,
                                                               const char *name) {
  (void)vfs;
  (void)name;
  return 0;
}

static const char *capstone_sqlite_vfs_next_system_call(sqlite3_vfs *vfs,
                                                        const char *name) {
  (void)vfs;
  (void)name;
  return 0;
}

void capstone_sqlite_vfs_bootstrap(struct capstone_sqlite_vfs_bundle *bundle) {
  if (!bundle)
    return;

  bundle->io.iVersion = 1;
  bundle->io.xClose = capstone_sqlite_io_close;
  bundle->io.xRead = capstone_sqlite_io_read;
  bundle->io.xWrite = capstone_sqlite_io_write;
  bundle->io.xTruncate = capstone_sqlite_io_truncate;
  bundle->io.xSync = capstone_sqlite_io_sync;
  bundle->io.xFileSize = capstone_sqlite_io_file_size;
  bundle->io.xLock = capstone_sqlite_io_lock;
  bundle->io.xUnlock = capstone_sqlite_io_unlock;
  bundle->io.xCheckReservedLock = capstone_sqlite_io_check_reserved_lock;
  bundle->io.xFileControl = capstone_sqlite_io_file_control;
  bundle->io.xSectorSize = capstone_sqlite_io_sector_size;
  bundle->io.xDeviceCharacteristics = capstone_sqlite_io_device_characteristics;
  bundle->io.xShmMap = 0;
  bundle->io.xShmLock = 0;
  bundle->io.xShmBarrier = 0;
  bundle->io.xShmUnmap = 0;
  bundle->io.xFetch = 0;
  bundle->io.xUnfetch = 0;

  bundle->vfs.iVersion = 3;
  bundle->vfs.szOsFile = (int)sizeof(struct capstone_sqlite_file);
  bundle->vfs.mxPathname = CAPSTONE_SQLITE_VFS_MAX_PATHNAME;
  bundle->vfs.pNext = 0;
  bundle->vfs.zName = CAPSTONE_SQLITE_VFS_NAME;
  bundle->vfs.pAppData = bundle;
  bundle->vfs.xOpen = capstone_sqlite_vfs_open;
  bundle->vfs.xDelete = capstone_sqlite_vfs_delete;
  bundle->vfs.xAccess = capstone_sqlite_vfs_access;
  bundle->vfs.xFullPathname = capstone_sqlite_vfs_full_pathname;
  bundle->vfs.xDlOpen = capstone_sqlite_vfs_dl_open;
  bundle->vfs.xDlError = capstone_sqlite_vfs_dl_error;
  bundle->vfs.xDlSym = capstone_sqlite_vfs_dl_sym;
  bundle->vfs.xDlClose = capstone_sqlite_vfs_dl_close;
  bundle->vfs.xRandomness = capstone_sqlite_vfs_randomness;
  bundle->vfs.xSleep = capstone_sqlite_vfs_sleep;
  bundle->vfs.xCurrentTime = capstone_sqlite_vfs_current_time;
  bundle->vfs.xGetLastError = capstone_sqlite_vfs_get_last_error;
  bundle->vfs.xCurrentTimeInt64 = capstone_sqlite_vfs_current_time_int64;
  bundle->vfs.xSetSystemCall = capstone_sqlite_vfs_set_system_call;
  bundle->vfs.xGetSystemCall = capstone_sqlite_vfs_get_system_call;
  bundle->vfs.xNextSystemCall = capstone_sqlite_vfs_next_system_call;
}

int capstone_sqlite_vfs_self_check(const struct capstone_sqlite_vfs_bundle *bundle) {
  if (!bundle)
    return CAPSTONE_SQLITE_VFS_CHECK_NULL;

  if (bundle->io.iVersion != 1)
    return CAPSTONE_SQLITE_VFS_CHECK_IO_VERSION;

  if (bundle->vfs.iVersion != 3)
    return CAPSTONE_SQLITE_VFS_CHECK_VFS_VERSION;

  if (bundle->vfs.szOsFile != (int)sizeof(struct capstone_sqlite_file))
    return CAPSTONE_SQLITE_VFS_CHECK_FILE_SIZE;

  if (bundle->vfs.pAppData != (void *)bundle)
    return CAPSTONE_SQLITE_VFS_CHECK_APPDATA;

  if (!bundle->vfs.zName || bundle->vfs.zName[0] == '\0')
    return CAPSTONE_SQLITE_VFS_CHECK_NAME;

  if (!bundle->io.xClose || !bundle->io.xRead || !bundle->io.xWrite ||
      !bundle->io.xTruncate || !bundle->io.xSync || !bundle->io.xFileSize ||
      !bundle->io.xLock || !bundle->io.xUnlock ||
      !bundle->io.xCheckReservedLock || !bundle->io.xFileControl ||
      !bundle->io.xSectorSize || !bundle->io.xDeviceCharacteristics)
    return CAPSTONE_SQLITE_VFS_CHECK_IO_POINTERS;

  if (!bundle->vfs.xOpen || !bundle->vfs.xDelete || !bundle->vfs.xAccess ||
      !bundle->vfs.xFullPathname || !bundle->vfs.xRandomness ||
      !bundle->vfs.xSleep || !bundle->vfs.xCurrentTime ||
      !bundle->vfs.xGetLastError || !bundle->vfs.xCurrentTimeInt64 ||
      !bundle->vfs.xSetSystemCall || !bundle->vfs.xGetSystemCall ||
      !bundle->vfs.xNextSystemCall)
    return CAPSTONE_SQLITE_VFS_CHECK_VFS_POINTERS;

  return CAPSTONE_SQLITE_VFS_CHECK_OK;
}


