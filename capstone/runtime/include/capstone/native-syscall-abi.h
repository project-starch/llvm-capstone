/* SPDX-License-Identifier: MIT */
#ifndef CAPSTONE_NATIVE_SYSCALL_ABI_H
#define CAPSTONE_NATIVE_SYSCALL_ABI_H
/* Experimental selector; not registered by the default virtual module. */
#define NS_SERVICE_NATIVE 0x43534e31
#define NS_ABI_VERSION 1
/* RV64 Linux POSIX-lock commands in the existing 32-byte flock wire ABI. */
#define NS_F_GETLK 5
#define NS_F_SETLK 6
#define NS_F_SETLKW 7
#endif
