obj-m += capstone_vm.o
# make CV_TEST_HOOKS=1: the test-only build used by run-step-hooks.py.
ifdef CV_TEST_HOOKS
ccflags-y += -DCV_TEST_HOOKS
endif
# Experimental transaction prototype; absent from ordinary adapter builds.
ifdef CV_GRANT_TXN
ccflags-y += -DCV_GRANT_TXN
endif

# make CV_RACE_TEST=1: the check-then-copy race experiment (tests/native-race).
ifdef CV_RACE_TEST
ccflags-y += -DCV_RACE_TEST
endif

ccflags-y += -I$(src)
