/* repro322_memfs — a whole in-memory FILESYSTEM for the freestanding corpus domains.
 *
 * WHY NOT ext/misc/memvfs.c, which 3.22.0 already ships:
 *
 *   1. It serves ONLY the main database: xOpen starts with
 *        if( (flags & SQLITE_OPEN_MAIN_DB)==0 ) return SQLITE_CANTOPEN;
 *      so the rollback journal cannot be created. Most upstream corrupt-database
 *      regression tests WRITE, and without a journal SQLite reports SQLITE_CORRUPT
 *      or SQLITE_FULL before reaching the defect. Measured: of 17 upstream corrupt
 *      images that crash 3.22.0 through a real file, memvfs reproduced 3.
 *   2. It takes the buffer as an INTEGER in the URI and casts it back to a pointer
 *        p->aData = (unsigned char*)sqlite3_uri_int64(zName,"ptr",0);
 *      On a capability machine that forges a pointer from an integer: the result is
 *      untagged and the first dereference faults. There is no way to recover a valid
 *      capability from an integer, by design.
 *
 * So this is a small filesystem instead: any name can be created, read, written,
 * truncated and deleted, which is all the pager needs from a journal. A database
 * image is preloaded with repro_memfs_register() and is thereafter an ordinary file.
 *
 * Storage is a dedicated static arena, NOT memsys5's: the corpus measures what
 * memsys5 does to the program under test, so the harness must not allocate from it.
 * No integer-to-pointer casts anywhere. Single-threaded: the locking methods are
 * honest no-ops because the domain has one thread. */
#include "sqlite3.h"

#ifndef REPRO_MEMFS_FILES
#define REPRO_MEMFS_FILES 8
#endif
#ifndef REPRO_MEMFS_ARENA
/* Keep this small: it is static BSS inside the domain image, and the loader
** refuses a domain whose loadable size is too large (create_dom failed, seen
** at 3.4 MB with a 768 KiB arena). The images are 24-40 KB and the journal is
** at most about the size of the pages it saves. */
#define REPRO_MEMFS_ARENA (160U * 1024U)
#endif
#define REPRO_MEMFS_NAME_MAX 64

static unsigned char memfs_arena[REPRO_MEMFS_ARENA] __attribute__((aligned(16)));
static unsigned long memfs_used;

struct memfs_node {
  char name[REPRO_MEMFS_NAME_MAX];
  unsigned char *data;
  sqlite3_int64 size;      /* bytes currently in the file */
  unsigned long cap;       /* bytes reserved for it */
  int inuse;
};
static struct memfs_node memfs_files[REPRO_MEMFS_FILES];

struct memfs_file { sqlite3_file base; struct memfs_node *node; };

static int memfs_streq(const char *a, const char *b){
  while(*a && *a == *b){ a++; b++; }
  return *a == *b;
}
static void memfs_strcpy(char *d, const char *s, int n){
  int i = 0; for(; s[i] && i < n-1; i++) d[i] = s[i]; d[i] = 0;
}
static void memfs_zero(unsigned char *p, unsigned long n){
  unsigned long i; for(i = 0; i < n; i++) p[i] = 0;
}
static void memfs_copy(unsigned char *d, const unsigned char *s, unsigned long n){
  unsigned long i; for(i = 0; i < n; i++) d[i] = s[i];
}

/* Carve a slice of the arena. Never freed: a domain runs once. */
static unsigned char *memfs_carve(unsigned long n){
  unsigned char *p;
  n = (n + 15UL) & ~15UL;
  if(memfs_used + n > sizeof(memfs_arena)) return 0;
  p = &memfs_arena[memfs_used];
  memfs_used += n;
  return p;
}

static struct memfs_node *memfs_find(const char *name){
  int i;
  for(i = 0; i < REPRO_MEMFS_FILES; i++)
    if(memfs_files[i].inuse && memfs_streq(memfs_files[i].name, name))
      return &memfs_files[i];
  return 0;
}

static struct memfs_node *memfs_create(const char *name, unsigned long cap){
  int i;
  for(i = 0; i < REPRO_MEMFS_FILES; i++){
    struct memfs_node *f = &memfs_files[i];
    if(f->inuse) continue;
    f->data = memfs_carve(cap);
    if(!f->data) return 0;
    memfs_zero(f->data, cap);
    memfs_strcpy(f->name, name, REPRO_MEMFS_NAME_MAX);
    f->size = 0; f->cap = cap; f->inuse = 1;
    return f;
  }
  return 0;
}

/* Preload a database image under a name. Call before sqlite3_open(). */
int repro_memfs_register(const char *name, const unsigned char *data, int n){
  struct memfs_node *f = memfs_create(name, (unsigned long)n + 16384UL);
  if(!f) return SQLITE_NOMEM;
  memfs_copy(f->data, data, (unsigned long)n);
  f->size = n;
  return SQLITE_OK;
}

static int memfsClose(sqlite3_file *p){ (void)p; return SQLITE_OK; }

static int memfsRead(sqlite3_file *p, void *buf, int amt, sqlite3_int64 off){
  struct memfs_file *f = (struct memfs_file *)p;
  unsigned char *out = (unsigned char *)buf;
  if(off >= f->node->size){ memfs_zero(out, (unsigned long)amt); return SQLITE_IOERR_SHORT_READ; }
  if(off + amt > f->node->size){
    long have = (long)(f->node->size - off);
    memfs_copy(out, &f->node->data[off], (unsigned long)have);
    memfs_zero(out + have, (unsigned long)(amt - have));
    return SQLITE_IOERR_SHORT_READ;
  }
  memfs_copy(out, &f->node->data[off], (unsigned long)amt);
  return SQLITE_OK;
}

static int memfsWrite(sqlite3_file *p, const void *buf, int amt, sqlite3_int64 off){
  struct memfs_file *f = (struct memfs_file *)p;
  if((unsigned long)(off + amt) > f->node->cap) return SQLITE_FULL;
  memfs_copy(&f->node->data[off], (const unsigned char *)buf, (unsigned long)amt);
  if(off + amt > f->node->size) f->node->size = off + amt;
  return SQLITE_OK;
}

static int memfsTruncate(sqlite3_file *p, sqlite3_int64 size){
  struct memfs_file *f = (struct memfs_file *)p;
  if(size < f->node->size) f->node->size = size;
  return SQLITE_OK;
}
static int memfsSync(sqlite3_file *p, int flags){ (void)p; (void)flags; return SQLITE_OK; }
static int memfsFileSize(sqlite3_file *p, sqlite3_int64 *pSize){
  *pSize = ((struct memfs_file *)p)->node->size; return SQLITE_OK;
}
static int memfsLock(sqlite3_file *p, int l){ (void)p; (void)l; return SQLITE_OK; }
static int memfsUnlock(sqlite3_file *p, int l){ (void)p; (void)l; return SQLITE_OK; }
static int memfsCheckReserved(sqlite3_file *p, int *r){ (void)p; *r = 0; return SQLITE_OK; }
static int memfsControl(sqlite3_file *p, int op, void *a){ (void)p; (void)op; (void)a; return SQLITE_NOTFOUND; }
static int memfsSectorSize(sqlite3_file *p){ (void)p; return 512; }
static int memfsDeviceChar(sqlite3_file *p){
  (void)p;
  /* Powersafe overwrite: the pager then skips some journal padding it would
   * otherwise need, and nothing here can tear a write. */
  return SQLITE_IOCAP_POWERSAFE_OVERWRITE;
}

static const sqlite3_io_methods memfs_io = {
  1, memfsClose, memfsRead, memfsWrite, memfsTruncate, memfsSync, memfsFileSize,
  memfsLock, memfsUnlock, memfsCheckReserved, memfsControl, memfsSectorSize, memfsDeviceChar,
  0, 0, 0, 0, 0, 0
};

static int memfsOpen(sqlite3_vfs *vfs, const char *name, sqlite3_file *p,
                     int flags, int *pOut){
  struct memfs_file *f = (struct memfs_file *)p;
  struct memfs_node *n;
  (void)vfs;
  if(!name) return SQLITE_CANTOPEN;          /* no transient files */
  n = memfs_find(name);
  if(!n){
    if((flags & SQLITE_OPEN_CREATE) == 0) return SQLITE_CANTOPEN;
    /* A journal is as big as the pages it saves; give it room. */
    n = memfs_create(name, 64UL * 1024UL);
    if(!n) return SQLITE_CANTOPEN;
  }
  f->base.pMethods = &memfs_io;
  f->node = n;
  if(pOut) *pOut = flags;
  return SQLITE_OK;
}

static int memfsDelete(sqlite3_vfs *vfs, const char *name, int sync){
  struct memfs_node *n = memfs_find(name);
  (void)vfs; (void)sync;
  if(n){ n->inuse = 0; n->size = 0; }       /* the slice stays carved; a domain runs once */
  return SQLITE_OK;
}
static int memfsAccess(sqlite3_vfs *vfs, const char *name, int flags, int *pOut){
  (void)vfs; (void)flags;
  *pOut = memfs_find(name) ? 1 : 0;
  return SQLITE_OK;
}
static int memfsFullPathname(sqlite3_vfs *vfs, const char *in, int n, char *out){
  (void)vfs; memfs_strcpy(out, in, n); return SQLITE_OK;
}
static void *memfsDlOpen(sqlite3_vfs *v, const char *z){ (void)v; (void)z; return 0; }
static void memfsDlError(sqlite3_vfs *v, int n, char *z){ (void)v; if(n>0) z[0]=0; }
static void (*memfsDlSym(sqlite3_vfs *v, void *h, const char *z))(void){ (void)v;(void)h;(void)z; return 0; }
static void memfsDlClose(sqlite3_vfs *v, void *h){ (void)v; (void)h; }
static int memfsRandomness(sqlite3_vfs *v, int n, char *z){
  int i; (void)v; for(i = 0; i < n; i++) z[i] = (char)(i * 37 + 11); return n;
}
static int memfsSleep(sqlite3_vfs *v, int micro){ (void)v; return micro; }
/* NO xCurrentTime. This port builds with -DSQLITE_OMIT_FLOATING_POINT=1, which sqliteInt.h
** implements as `#define double sqlite_int64` -- but sqlite3.h declared xCurrentTime with a
** REAL double before that, so a function written here with `double*` has a different type
** from the slot and the initializer is rejected. Declaring the VFS iVersion 2 and supplying
** only xCurrentTimeInt64 avoids naming `double` at all; sqlite3OsCurrentTimeInt64 prefers it
** and nothing else calls xCurrentTime. */
static int memfsCurrentTimeInt64(sqlite3_vfs *v, sqlite3_int64 *p){
  (void)v; *p = 24405875LL * 8640000LL; return SQLITE_OK;
}

static sqlite3_vfs memfs_vfs = {
  2, (int)sizeof(struct memfs_file), REPRO_MEMFS_NAME_MAX, 0, "repro_memfs", 0,
  memfsOpen, memfsDelete, memfsAccess, memfsFullPathname,
  memfsDlOpen, memfsDlError, memfsDlSym, memfsDlClose,
  memfsRandomness, memfsSleep, 0 /* xCurrentTime: see above */, 0,
  memfsCurrentTimeInt64, 0, 0, 0
};

int repro_memfs_init(void){ return sqlite3_vfs_register(&memfs_vfs, 1); }
