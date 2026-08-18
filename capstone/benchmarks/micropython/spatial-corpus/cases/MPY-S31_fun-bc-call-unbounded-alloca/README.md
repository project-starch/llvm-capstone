# MPY-S31: unbounded alloca when the VM state allocation fails

Source: #19129, https://github.com/micropython/micropython/issues/19129 (OPEN, so
present at the pin and needing no parent build).

`fun_bc_call` falls back to `alloca` when `m_new_obj_var_maybe` fails for the VM
state, instead of raising. Reachable only when `MICROPY_ENABLE_PYSTACK` is 0, which
is this port's setting.

Two arms, one image, one boot, ordered so the arm expected to return goes first:

- `01_s31_alloca_taken.py` — is the fallback taken at all?
- `02_s31_alloca_recursive.py` — which guard fires first, the port's or the hardware's?

Both compile their function **before** filling the heap. That is the whole trick:
an earlier shape built `str(list(range(N)))` at runtime and the MemoryError arrived
while building the source, so the call was never reached and the ladder ran to
completion having tested nothing.

See RESULT.txt. The short version: the fallback runs untrapped, and the port's own
`mp_cstack_check` stops recursion at depth 8, well before the stack capability's
bound — so this row cannot demonstrate a hardware stack trap and the corpus says so.
