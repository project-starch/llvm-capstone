MODULE_big = pgcorpus_lifetime
OBJS = \
	$(WIN32RES) \
	pgcorpus_lifetime.o

EXTENSION = pgcorpus_lifetime
DATA = pgcorpus_lifetime--1.0.sql
PGFILEDESC = "pgcorpus_lifetime - parent, sibling and successor controls at the memory contexts"

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
subdir = contrib/pgcorpus_lifetime
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif
