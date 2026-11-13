obj-m += capstone_vm.o
# make CV_TEST_HOOKS=1: the test-only build used by run-step-hooks.py.
ifdef CV_TEST_HOOKS
ccflags-y += -DCV_TEST_HOOKS
endif
# make CAPSTONE_VM_IDS=63: the processor's lifetime-id width (30 by default, 55 or 63; cv-format.h).
ifdef CAPSTONE_VM_IDS
ccflags-y += -DCAPSTONE_VM_IDS=$(CAPSTONE_VM_IDS)
endif
# Experimental transaction prototype; absent from ordinary adapter builds.
ifdef CV_GRANT_TXN
ccflags-y += -DCV_GRANT_TXN
endif
