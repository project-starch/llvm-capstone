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

ccflags-y += -I$(src)
