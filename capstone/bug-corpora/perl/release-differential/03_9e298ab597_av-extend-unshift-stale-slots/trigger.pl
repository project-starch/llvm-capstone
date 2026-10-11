require "shim.pl";
# upstream t/op/array.t hunk from 9e298ab597 ("Perl_av_extend_guts: Zero()
# trailing elements after unshift & resize").  Adapted: the upstream test is
# fresh_perl_is('my @x;$x[0] = 1;shift @x;$x[22] = 1;$x[25] = 1;', ...), whose
# program body is run here in-process (the shim's fresh_perl_is is a stub).
# GH #21235
{
    my @x;
    $x[0] = 1;
    shift @x;
    $x[22] = 1;
    $x[25] = 1;
    pass('unshifting and growing an array initializes trailing elements');
}
