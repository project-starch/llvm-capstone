# Static capability globals probe
This directory holds a **small permanent runtime diagnostic** for the current
LLVM-generated-domain blocker around static/global objects with capability-valued
fields.
## Variants
- `direct_use_domain.c`
  - control case,
  - uses the same logical values directly in code,
  - currently expected to execute successfully.
- `runtime_materialize_domain.c`
  - positive runtime-side materialization POC,
  - populates a writable global object at runtime and then uses it,
  - currently expected to execute successfully.
- `static_const_domain.c`
  - reduced reproducer,
  - stores those values in a file-scope `static const` object and then loads them
    back at runtime,
  - currently expected to trigger the existing capability failure.
## Expected current behavior
The control case should print:
```text
Called dom (1-th time) retval = 305397871
```
The runtime-materialization POC should currently print the same value:
```text
Called dom (1-th time) retval = 305397871
```
The reproducer case is currently expected to fail with:
```text
[CAPSTONE] cs.cjalr requires capability in rs1
```
## Purpose
This probe is intended as a stable regression/diagnostic target while the tree
works toward runtime-side support for static/global capability-bearing objects.
It is **not** yet part of the validated baseline smoke bundle.
## Related files
- `metadata_contract.h`
  - minimal shared descriptor shape for runtime-side materialization,
  - intended to stay useful whether the policy becomes eager-at-init or lazy-on-first-use.
- `metadata_contract.md`
  - maps the current reduced failing case onto that minimal descriptor model.
