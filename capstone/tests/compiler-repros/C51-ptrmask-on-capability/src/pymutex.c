/* The shape CPython 3.13 reaches in 26 objects: PyMutex is one byte, locked by
   compare-exchange (Include/cpython/lock.h, PyMutex_Lock). */
#include <stdint.h>
typedef struct { uint8_t _bits; } PyMutex;
void PyMutex_LockSlow(PyMutex *m);
void PyMutex_Lock(PyMutex *m)
{
    uint8_t expected = 0;
    if (!__atomic_compare_exchange_n(&m->_bits, &expected, 1, 0,
                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
        PyMutex_LockSlow(m);
}
