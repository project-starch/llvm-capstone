ifneq ($(CONFIG_RISCV_CAPSTONE_TASK),y)
$(error The kernel's Sublet code requires built-in CONFIG_RISCV_CAPSTONE_TASK=y)
endif
ifneq ($(KBUILD_EXTMOD),)
$(error The kernel's Sublet code is built in; use install-native-task.sh for an in-tree build)
endif
obj-y += capstone_vm.o
ccflags-y += -I$(src)
