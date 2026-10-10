require "shim.pl";
# upstream t/op/state.t has the state/lexical_subs features enabled file-wide;
# re-enable them here so the eval STRING below parses standalone.
use feature qw(state lexical_subs);
# upstream t/op/state.t hunk from 17535c984a ("fix refcount on cloned constant
# state subs").  Verbatim.
# This caused 'Attempt to free unreferenced scalar' because the SV holding
# the value of the const state sub wasn't having its ref count incremented
# when the sub was cloned.
{
    my @warnings;
    local $SIG{__WARN__} = sub { push @warnings, @_ };
    my $e = eval 'my $s = sub { state sub FOO () { 42 } }; 1;';
    is($e, 1, "const state sub ran ok");
    ok(!@warnings, "no 'Attempt to free unreferenced scalar'");
}
