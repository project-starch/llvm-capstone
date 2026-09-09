#!/usr/bin/env bash
# MPY-T07 / upstream 4128: lex->source_name read after mp_parse freed the lexer.
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# 0x70000001: the read completed and the freed block still held its old value --
# untrapped, staleness NOT demonstrated. A RECONSTRUCTION, not the upstream program.

repro_glue MPY_T07_LEXER_UAF mpy_t07 1879048193
