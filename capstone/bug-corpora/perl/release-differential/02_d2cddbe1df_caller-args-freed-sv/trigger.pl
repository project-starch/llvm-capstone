require "shim.pl";
# upstream t/op/caller.t hunk from d2cddbe1df ("caller(): don't copy freed SV
# pointer to @DB::args").  Adapted: ::is -> is (no test.pl package games).
{
    # Try to avoid copying pointers to freed SVs into @DB::args.
    # previously this caused "panic: attempt to copy freed scalar"
    my @a = 'A';
    sub {
        my $i = shift;
        my $j = shift;
        @a = (); # free the 'A' scalar
        package DB;
        () = caller(0);
        my $x = $DB::args[0];
        my $y = $DB::args[1];
        main::is("$x-$y", "-B", "no freed scalars");
    }
    ->($a[0], 'B');
}
