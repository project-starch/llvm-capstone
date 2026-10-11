require "shim.pl";
# upstream t/op/for.t hunk from ad5fedb002 ("for loops: protect GV/LVREF from
# premature free").  Verbatim.
# the GV of the loop variable didn't have its refcount bumped while being
# used by the loop, so it was possible to free it mid-loop.  This used to
# assert/SEGV
{
    my $f = "a_low_refcnt_package_var";
    my $i = 0;
    no strict 'refs';
    for ${*$f} (5,11,33) {
        delete $main::{$f};
        $i++;
    }
    is($i, 3, "deleting glob is safe");
}
