obj-m += capstone_vm.o
# make CV_TEST_HOOKS=1: the test-only build used by run-step-hooks.py.
ifdef CV_TEST_HOOKS
ccflags-y += -DCV_TEST_HOOKS
endif
# make CV_FLOOR=1: EXPERIMENT build, floor=1 by default (no payload pins, no tag sweep). Never lands.
ifdef CV_FLOOR
ccflags-y += -DCV_FLOOR
endif
# make CV_UNPINNED=1: EXPERIMENT build for a tag-aware guest Linux (linux experiment/tag-aware-mm):
# floor + faults on every arena resolved by the kernel's fault path. Never lands.
ifdef CV_UNPINNED
ccflags-y += -DCV_UNPINNED
endif
