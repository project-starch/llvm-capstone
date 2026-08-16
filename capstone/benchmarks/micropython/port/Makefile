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
CFLAGS += $(INC) -std=c99 $(COPT) -O0
LIBS =

# main.c only exists so mkrules.mk has a source to scan for qstrs alongside the core.
SRC_C = main.c

OBJ += $(PY_CORE_O)
OBJ += $(addprefix $(BUILD)/, $(SRC_C:.c=.o))

# All four generated headers, not just the qstrs: mpstate.h includes root_pointers.h,
# runtime includes moduledefs.h, and ROM text compression adds compressed.data.h.
# mkrules.mk lists them as OBJ_EXTRA_ORDER_DEPS, which only fires when an object is
# built -- and this Makefile builds none, so they are named explicitly here.
all: $(BUILD)/genhdr/qstrdefs.generated.h $(HEADER_BUILD)/moduledefs.h \
     $(HEADER_BUILD)/root_pointers.h $(HEADER_BUILD)/compressed.data.h

include $(TOP)/py/mkrules.mk
