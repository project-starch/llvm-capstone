# Header generation only. The real build is build-micropython-silicon.sh, which compiles the
# whole core as ONE translation unit for capstone64; this Makefile exists because the qstr,
# module and root-pointer headers are generated from mpconfigport.h and are therefore
# port-specific. Borrowing ports/minimal's headers does not work: MICROPY_STACK_CHECK alone
# pulls in a qstr that port does not define, and a mismatched string pool is the kind of error
# that compiles.
#
# The build script copies this directory into the MicroPython tree as ports/capstone and runs
# `make -C ports/capstone build/genhdr/qstrdefs.generated.h` with a STOCK compiler.
include ../../py/mkenv.mk

CROSS = 0
QSTR_DEFS = qstrdefsport.h
MICROPY_ROM_TEXT_COMPRESSION ?= 1

include $(TOP)/py/py.mk

INC += -I. -I$(TOP) -I$(BUILD)
LD = $(CC)
# Keep header generation under the same feature defines as the domain compilation. In
# particular, qstr/module discovery is configuration-dependent, so a minimal header pool cannot
# be reused by a CORE_FEATURES or EXTRA_FEATURES image.
CFLAGS += $(INC) -std=c99 $(COPT) -O0 $(CFLAGS_EXTRA)
LIBS =

# main.c only exists so mkrules.mk has a source to scan for qstrs alongside the core.
SRC_C = main.c

# The extmod modules this port carries. They must be scanned here, not merely compiled by
# build-micropython-silicon.sh: MP_REGISTER_MODULE only reaches genhdr/moduledefs.h through the
# qstr pass, so a module compiled but not scanned links fine and is simply absent at runtime.
# build-micropython-silicon.sh parses this exact list, so edit it here and nowhere else.
EXTMOD_SRC_C = \
	extmod/modbinascii.c \
	extmod/moddeflate.c \
	extmod/modframebuf.c \
	extmod/modhashlib.c \
	extmod/modheapq.c \
	extmod/modjson.c \
	extmod/modrandom.c \
	extmod/modre.c \
	extmod/moductypes.c \

# The filesystem stack. Only reached when MPY_VFS=1 puts MICROPY_VFS in the defines:
# each file is wrapped in its own `#if MICROPY_VFS`, so listing them here costs
# nothing in a default build and keeps header generation in step with the amalgam.
EXTMOD_SRC_C += \
	extmod/modvfs.c \
	extmod/vfs.c \
	extmod/vfs_blockdev.c \
	extmod/vfs_fat.c \
	extmod/vfs_fat_diskio.c \
	extmod/vfs_fat_file.c \
	extmod/vfs_reader.c \

SRC_QSTR += $(addprefix $(TOP)/, $(EXTMOD_SRC_C))

OBJ += $(PY_CORE_O)
OBJ += $(addprefix $(BUILD)/, $(SRC_C:.c=.o))

# All four generated headers, not just the qstrs: mpstate.h includes root_pointers.h,
# runtime includes moduledefs.h, and ROM text compression adds compressed.data.h.
# mkrules.mk lists them as OBJ_EXTRA_ORDER_DEPS, which only fires when an object is
# built -- and this Makefile builds none, so they are named explicitly here.
all: $(BUILD)/genhdr/qstrdefs.generated.h $(HEADER_BUILD)/moduledefs.h \
     $(HEADER_BUILD)/root_pointers.h $(HEADER_BUILD)/compressed.data.h

include $(TOP)/py/mkrules.mk
