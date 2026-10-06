MODULE_big = pgcorpus_reach
OBJS = \
	$(WIN32RES) \
	pgcorpus_reach.o

# pgcrypto's headers only, for the real PGP_Context that case 09 turns on.
# No pgcrypto object is linked: it cannot be built for a Capstone domain.
PG_CPPFLAGS = -I$(top_srcdir)/contrib/pgcrypto

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
