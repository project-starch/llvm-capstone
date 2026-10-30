#!/bin/bash
# OB_BOOK / SB_BOOK for the dry run: no hub call. "book" prints a booking id, "unbook" succeeds.
[ "$1" = book ] && echo 999
exit 0
