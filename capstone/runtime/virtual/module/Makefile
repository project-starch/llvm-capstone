obj-m += capstone_vm.o
# make CV_TEST_HOOKS=1: the test-only build used by run-step-hooks.py.
ifdef CV_TEST_HOOKS
ccflags-y += -DCV_TEST_HOOKS
endif
# make CV_FLOOR=1: EXPERIMENT build, floor=1 by default (no payload pins, no tag sweep). Never lands.
ifdef CV_FLOOR
ccflags-y += -DCV_FLOOR
endif
