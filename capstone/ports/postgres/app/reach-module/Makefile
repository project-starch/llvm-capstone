MODULE_big = pgcorpus_reach
OBJS = \
	$(WIN32RES) \
	pgcorpus_reach.o

EXTENSION = pgcorpus_reach
DATA = pgcorpus_reach--1.0.sql
PGFILEDESC = "pgcorpus_reach - C-level callers for defects SQL cannot reach"

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
subdir = contrib/pgcorpus_reach
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif
