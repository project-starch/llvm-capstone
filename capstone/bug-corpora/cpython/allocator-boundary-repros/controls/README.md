# Controls for the virtual arms

Not cases. Two triggers in the corpus's own format, run on the same interpreter image as the cases
before any case runs. What each must do is the configuration's, in `tools/arms.json`:

| control | the access | virtual-cpython | virtual-cpython-pools |
|---|---|---|---|
| `control_uaf_block.py` | a freed pymalloc block written and read (`check_pyobject_freed_is_freed`) | complete | fault |
| `control_bounds_block.py` | a read past a pymalloc block's end (`check_pyobject_forbidden_bytes_is_freed`) | complete | fault |

A fault counts for a control only when its pc lies in the control's function or in what that
function inlines (`test_pyobject_is_freed`, `_PyObject_IsFreed`). Both come from
`_testinternalcapi`, which the image carries because it is built with `CPY_TEST_CAPI=1`.

Neither control reaches memory that mallocng owns directly, so neither shows that the system
allocator itself catches an access to a libc-malloc object. Of this corpus's rows, those are
the non-nested ones.
