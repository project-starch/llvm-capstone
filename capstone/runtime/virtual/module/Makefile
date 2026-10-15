obj-m += capstone_vm.o
# make CV_TEST_HOOKS=1: the test-only build used by run-step-hooks.py.
ifdef CV_TEST_HOOKS
ccflags-y += -DCV_TEST_HOOKS
endif
