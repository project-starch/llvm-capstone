/* The survey's MUST_FAIL control, compiled by survey-cpython-capstone.py
 * through capstone-cc exactly as CPython's own objects are, with CPython's core
 * flags.
 *
 * It states the one fact that makes this port a port: no integer type is as
 * wide as a pointer, because a pointer is a capability. That fails to compile
 * for capstone64 and compiles everywhere else. So a survey whose control
 * compiles was not building for a capability target at all, or its compiler
 * or its log reads every result as a success, and its counts mean nothing.
 *
 * It replaced Objects/longobject.o in the role (2026-09-23), which failed on
 * the same fact until patch 0007 decided what PyLong_FromVoidPtr means here.
 * A real object that must fail forever is not something a finished port has.
 */
#include "Python.h"

_Static_assert(sizeof(long long) >= sizeof(void *),
               "no integer type holds a pointer: a pointer is a capability");
