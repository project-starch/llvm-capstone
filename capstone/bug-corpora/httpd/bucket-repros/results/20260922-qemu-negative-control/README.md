# Negative control for the 2026-09-22 QEMU record

Cases 0, 4 and 6, both arms, with the fixture's event count set to 2 while
the file carries one event: the program's own CHECK refuses the input before
any pool is created, so no case runs. Every oracle must report FAIL -- the
spatial one because nothing completed, the sublet one because nothing
faulted at the published probe (case 4's completion oracle, because nothing
completed). All six did. A suite whose oracles cannot say FAIL proves nothing
by saying PASS; this is the record that they can.
