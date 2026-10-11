require "shim.pl";
# upstream t/op/eval.t hunk from b7b77ffc1e ("CLEAR_ERRSV: create a new SV if
# the existing one isGV_with_GP"), GH #16885.  Adapted: upstream runs the
# one-liner via fresh_perl_is('for$@(*0){eval}', '', undef, ...); the shim's
# fresh_perl_is is a stub, so the program body is executed in-process here.
# The second statement is an addition: CLEAR_ERRSV's SvPVCLEAR() on the glob
# runs sv_grow(), which overwrites sv_u.svu_gp (i.e. GvGP) with a fresh 1-byte
# malloc'ed buffer.  The illegal access happens the next time a glob slot is
# read through that pointer, so read $0 (the SV slot of *0) back; this SEGVs on
# an unfixed perl.
for $@ (*0) { eval }
my $v = $0;
pass('GH #16885 - isGV_with_GP(PL_errgv)');
