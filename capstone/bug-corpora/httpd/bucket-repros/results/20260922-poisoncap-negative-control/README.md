# Negative control for the 2026-09-22 PoisonCap record

Cases 0 and 4, both modes, with the fixture's event count set to 2 while the
file carries one event: the program refuses the input before any pool is
created and exits 3, so no case runs. Every oracle must report FAIL -- mode 0
because nothing completed and no report was written, mode 1 because nothing
faulted at the resolved probe. All four did; `arms_passed: 0` is the control
firing. A suite whose oracles cannot say FAIL proves nothing by saying PASS.
