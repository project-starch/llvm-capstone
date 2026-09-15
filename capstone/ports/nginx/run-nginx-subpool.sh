#!/usr/bin/env bash
# Does the level below keep its promise? port/ngx_subpool_test.c asks, and this runs it.
#
# It exists because the invocation is not guessable and both ways of guessing it wrong produce
# something that looks like a broken port:
#
#   NGX_ARENA_LINEAR=1  the guest shares the region REV_TRANSFERRED instead of REV_SHARED, so the
#                       arena arrives LINEAR. Without it the arena arrives NONLIN, csmrev refuses
#                       it, and the driver stops at its own first guard with 0xFD0001. That guard
#                       is the only reason this reads as "the arena is the wrong type" rather than
#                       as a fault three calls later.
#   NGX_SUBLET unset    the flag that patches ngx_palloc.c. Setting it here includes
#                       port/ngx_subpool.c a second time and the amalgam does not compile. The
#                       level below's own test does not want the level above at all.
#
# THE EXPECTATION IS THE WHOLE MARK AND NOT JUST THE FAILURE COUNT. The driver reports the phase it
# reached in bits 16..23, and a build that silently kept an older image reported a plausible count
# and no failures three times in a row here. A phase that does not reach its last value says so.
#
#   phase 14, 62 checks, 0 failures -> 0x0E3E00
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

NGX_DOMAIN=subpool NGX_ARENA_LINEAR=1 DOM_NAME=ngx-subpool \
    NGX_EXPECT_MARK=$((16#0E3E00)) bash "$SCRIPT_DIR/run-nginx-domain.sh"
