#!/bin/bash
# OB_FLASH for the dry run: no console call. Prints its argv and exits with $OB_FLASH_RC.
echo "FLASH-STUB name=$1 file=$2 sha=${3:0:16}"
exit ${OB_FLASH_RC:-0}
