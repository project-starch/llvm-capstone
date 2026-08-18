# MPY-S01: Integer overflow in sequence repeat causes heap buffer overflow

Source: #19314, https://github.com/micropython/micropython/issues/19314
State at fetch: CLOSED. Present at the pin regardless -- see RESULT.txt.

`seq * n` sizes the result as `len(seq) * n` in unchecked `size_t`, allocates
that (wrapped) size, then writes the real element count. Affects `str`, `bytes`,
`list` and `tuple`, i.e. `py/objstr.c`, `py/objlist.c`, `py/objtuple.c` and
`py/sequence.c`.

`01_s01_bytes_repeat.py`, `02_s01_list_repeat.py` and `03_s01_tuple_repeat.py`
are the three arms actually run in the domain; `00_sanity.py` is the control that
makes them readable. All four are baked into one image and run from one boot by
`tools/run-resumable-suite.py`, which reboots after each fault and continues.

    MPY_TESTS=all MPY_TEST_BASE_DIR=capstone-spatial MPY_FLOAT_CORE=1 \
    MPY_TEST_INCLUDE_UNSUPPORTED=1 \
    DOMAIN_EXTRA_DEFS="-DMICROPY_CONFIG_ROM_LEVEL=MICROPY_CONFIG_ROM_LEVEL_EXTRA_FEATURES" \
    DOM_NAME=mpy_spatial_suite bash ../../build-micropython-silicon.sh

`MPY_TEST_INCLUDE_UNSUPPORTED=1` is load-bearing and not a convenience: the table
generator derives each expectation by running the test on the host, and drops any
test that exits non-zero. Every arm here is supposed to crash the host, so without
that flag the image was built with three of the four tests SILENTLY MISSING and
reported "1 tests kept, 3 skipped" in a line easy to read past.

The guest runner must be `tools/mpy-resume-guest.c`, not `capstone-test.user`.
See RESULT.txt for what happens when it is not.
