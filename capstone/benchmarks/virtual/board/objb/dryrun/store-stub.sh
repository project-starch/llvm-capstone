#!/bin/bash
# OB_STORE for the dry run: rc 1 for a name the store does not hold, 0 for bitstream 3. No console call.
echo "STORE-STUB $*"
case $1 in caplifive_vp2c_e7af48457.bit) exit 0;; *) exit 1;; esac
