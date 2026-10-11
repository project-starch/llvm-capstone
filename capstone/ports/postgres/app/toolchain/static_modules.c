/*
 * The loadable modules a domain needs, linked into the image (patch 0015).
 *
 * A domain cannot dlopen: there is no dynamic loader and no second image. The
 * setup SQL initdb runs needs two modules all the same, dict_snowball (its text
 * search dictionaries) and plpgsql (CREATE EXTENSION plpgsql), and a backend
 * resolves a C function by loading its library, even to validate CREATE
 * FUNCTION. So build-domain.sh compiles each module's objects into the image,
 * with the two names every module defines (Pg_magic_func and _PG_init) renamed
 * <module>_Pg_magic_func and <module>__PG_init, and writes
 * static_modules_table.h: per module a PGSU_MODULE(name) line, one
 * PGSU_SYMBOL(module, "name", symbol) line per global function it defines, and
 * PGSU_MODULE_END.
 *
 * dfmgr.c then calls these in place of stat, dlopen, dlsym and dlerror. A
 * library is found by its base name without DLSUFFIX, whatever directory it was
 * asked for in ("$libdir/plpgsql" and "plpgsql" are the same module); any other
 * file goes to the real stat and fails to load, as it would natively without
 * the file.
 */
#include <errno.h>
#include <string.h>
#include <sys/stat.h>

struct pgsu_symbol {
  const char *name;
  void *addr;
};
struct pgsu_module {
  const char *name;
  const struct pgsu_symbol *symbols;
};

/* The table's declarations: every symbol as a function of no arguments, since
 * only its address is taken here and dfmgr.c casts it back to the type the
 * caller expects. */
#define PGSU_MODULE(m)
#define PGSU_SYMBOL(m, name, sym) extern void sym(void);
#define PGSU_MODULE_END
#include "static_modules_table.h"
#undef PGSU_MODULE
#undef PGSU_SYMBOL
#undef PGSU_MODULE_END

/* One symbol array per module, closed by a NULL name. */
#define PGSU_MODULE(m) static const struct pgsu_symbol pgsu_syms_##m[] = {
#define PGSU_SYMBOL(m, name, sym) {name, (void *)sym},
#define PGSU_MODULE_END {0, 0}};
#include "static_modules_table.h"
#undef PGSU_MODULE
#undef PGSU_SYMBOL
#undef PGSU_MODULE_END

#define PGSU_MODULE(m) {#m, pgsu_syms_##m},
#define PGSU_SYMBOL(m, name, sym)
#define PGSU_MODULE_END
static const struct pgsu_module pgsu_modules[] = {
#include "static_modules_table.h"
};
#undef PGSU_MODULE
#undef PGSU_SYMBOL
#undef PGSU_MODULE_END

#define PGSU_NMODULES (sizeof(pgsu_modules) / sizeof(pgsu_modules[0]))

static const char *pgsu_error;

/* The module a file name means, or NULL: its last component, less ".so". */
static const struct pgsu_module *pgsu_find(const char *path) {
  const char *base = strrchr(path, '/');
  size_t len;
  base = base ? base + 1 : path;
  len = strlen(base);
  if (len > 3 && strcmp(base + len - 3, ".so") == 0)
    len -= 3;
  for (unsigned i = 0; i < PGSU_NMODULES; i++)
    if (strlen(pgsu_modules[i].name) == len &&
        strncmp(pgsu_modules[i].name, base, len) == 0)
      return &pgsu_modules[i];
  return 0;
}

/* dfmgr.c tells libraries apart by (st_dev, st_ino); a linked-in module gets
 * device 0 and its table index plus one, which no file on the share has. */
int pgsu_module_stat(const char *path, struct stat *buf) {
  const struct pgsu_module *m = pgsu_find(path);
  if (!m)
    return stat(path, buf);
  memset(buf, 0, sizeof(*buf));
  buf->st_dev = 0;
  buf->st_ino = (ino_t)(m - pgsu_modules) + 1;
  buf->st_mode = S_IFREG | 0444;
  return 0;
}

void *pgsu_module_open(const char *path) {
  const struct pgsu_module *m = pgsu_find(path);
  if (!m) {
    pgsu_error = "not linked into this domain image (a domain cannot load a library)";
    return 0;
  }
  return (void *)m;
}

void *pgsu_module_sym(void *handle, const char *name) {
  const struct pgsu_module *m = handle;
  for (const struct pgsu_symbol *s = m->symbols; s->name; s++)
    if (strcmp(s->name, name) == 0)
      return s->addr;
  return 0;
}

char *pgsu_module_error(void) {
  const char *e = pgsu_error;
  pgsu_error = 0;
  return (char *)e;
}
