#!/bin/bash
# SB_BOOK for the dry run: no hub call. "book" prints a booking id, "unbook" succeeds.
echo "BOOK-STUB $*" >> "${PLAIN_DRY:?}/book.log"
[ "$1" = book ] && echo 999
exit 0
