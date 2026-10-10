require "shim.pl";
# upstream t/io/scalar.t hunk from 254b30e378 ("PerlIOScalar_write: handle the
# case where vbuf overlaps the target scalar").  Verbatim.
{
    open my $fh, '>', \my $str or die $!;
    # Needs a sufficiently long string to trigger string expansion (sv_grow).
    print $fh "abcdefghijklmnopqrstuvwxyz";
    print $fh $str;
    is($str, "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
       "write a string to itself");
}
