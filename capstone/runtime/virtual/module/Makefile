ifneq ($(CONFIG_RISCV_CAPSTONE_TASK),y)
$(error Capstone requires built-in CONFIG_RISCV_CAPSTONE_TASK=y)
endif
ifneq ($(KBUILD_EXTMOD),)
$(error Capstone is native-only; use install-native-task.sh for an in-tree build)
endif
obj-y += capstone_vm.o
# make CV_TEST_HOOKS=1: the test-only build used by run-step-hooks.py.
ifdef CV_TEST_HOOKS
ccflags-y += -DCV_TEST_HOOKS
endif
# The transaction prototype depended on the retired device ioctl entry.
ifdef CV_GRANT_TXN
$(error CV_GRANT_TXN requires the historical ioctl adapter)
endif

# make CV_RACE_TEST=1: the check-then-copy race experiment (tests/native-race).
ifdef CV_RACE_TEST
ccflags-y += -DCV_RACE_TEST
endif

# make CV_FLAT_INDEX_BITS=16|18 (MEASUREMENT ONLY, with capstone_vm.c5a_model=1): the flat RECYCLE lifetime
# table of the boards' model, {generation, index} ids with that many index bits (16: QEMU's flat format,
# 18: C.5a and its QEMU twin), instead of the paged table.
ifdef CV_FLAT_INDEX_BITS
ccflags-y += -DCAPSTONE_VM_FLAT_TABLE=1 -DCAPSTONE_VM_INDEX_BITS=$(CV_FLAT_INDEX_BITS)
endif
# make CV_FLAT_INDEX_BITS=25 CV_IDS=55 (MEASUREMENT ONLY): option B's 55-bit lifetime id (capstone-ariane rtl-ids55,
# {generation[54:25], slot[24:0]}), flat wide records. Needs the ids-55 cap_rev_table_abi.h (CAP_REV_WIDE_*) beside the
# module: stage it from a capstone-qemu tree that has x-capstone-ids=55 (CAPSTONE_QEMU_SOURCE).
ifdef CV_IDS
ccflags-y += -DCAPSTONE_VM_IDS=$(CV_IDS)
endif
ccflags-y += -I$(src)
