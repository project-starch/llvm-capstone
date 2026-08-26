# MPY-S05: buffer overflow after a failed array grow

Source: #15271, https://github.com/micropython/micropython/pull/15271

`array_append` grows by eight slots when `self->free` hits zero. Before
3d93fed0aab8 it set `self->free = 8` and only then called `m_renew`; if that
raised MemoryError the count stayed at eight with the buffer unchanged, and the
next append wrote past the end.

`revert-the-fix.patch` puts the defect back on the pinned tree. Apply it to the
MicroPython working tree, build with `MPY_TEST_BASE_DIR` pointing at a directory
holding the two scripts here, and run with `tools/run-resumable-suite.py`:

    MPY_TESTS=all MPY_TEST_BASE_DIR=<dir> MPY_TEST_INCLUDE_UNSUPPORTED=1 \
    MPY_FLOAT_CORE=1 \
    DOMAIN_EXTRA_DEFS="-DMICROPY_CONFIG_ROM_LEVEL=MICROPY_CONFIG_ROM_LEVEL_EXTRA_FEATURES" \
    DOM_NAME=mpy_s05 bash ../../../build-micropython-silicon.sh

**Reverse the patch afterwards.** The tree under `$CAPSTONE_TMP_ROOT/micropython`
is shared by every other build, and a stale revert there would put this defect
into images that are supposed to be stock.

The first field of the output is the precondition flag, and it is the reason this
case has a RESULT worth reading: four earlier shapes printed a plausible line
having never triggered anything. See RESULT.txt.
