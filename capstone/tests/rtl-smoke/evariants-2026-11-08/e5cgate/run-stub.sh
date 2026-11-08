#!/bin/bash
# dry-run stub for run_sup_bare.py: writes the control set's uart.txt for SUP_IMG's image into SUP_OUT
n=$(basename "$SUP_IMG" .bin); mkdir -p "$SUP_OUT"; cp "$E0_DRY_SRC/$n/uart.txt" "$SUP_OUT/uart.txt"
[ "${E0_DRY_HARD:-}" = "$n" ] && echo "HARD STOP: resident is not the expected bitstream" && exit 1
echo "stub ran $n with FPGA_BITSTREAM=$FPGA_BITSTREAM rec=$SUP_REC_ADDR"; exit 0
