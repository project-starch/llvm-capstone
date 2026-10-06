require "shim.pl";
# forward declarations so the upstream bareword-style calls (like "B", qr//, ...)
# parse; the shim defines these subs at require time.
sub like; sub unlike; sub is; sub ok; sub pass; sub fail;
# upstream t/re/pat.t hunk from 40727c420c ("pat codeblocks: add size to struct
# reg_code_blocks"), GH #16627.  Adapted: (a) the reduced case from the commit
# message is run first, because at the pin the later statements croak
# "Eval-group not allowed" once the code-block bookkeeping has already gone
# wrong; (b) the upstream block is wrapped in eval{} so the whole body is
# reached even when one of those croaks fires.

# The reduced case from the commit message: "Here the B code block is stored in
# pRExC_state->code_blocks->cb[0], but then S_concat_pat() recurses, n is reset
# to 0, and the A code block is also stored into slot 0. Then things would
# start to crash."
{
    use re 'eval';
    my @x = ( qr/(?{ 'A' })/ );
    eval { my $p = qr/(?{ 'B' })@x/; "ab" =~ $p; 1 };
    pass("GH #16627 reduced case");
}

eval {
    # adaptation: upstream t/re/pat.t enables "use re 'eval'" only partway
    # through this block; hoist it so a croak from an earlier statement does not
    # abort the whole eval and cut the test body short.
    use re 'eval';
    # Various tests for regexes with code blocks interpolated from an
    # array, related to fixing GH #16627.
    #
    # Prior to the fix, some of these tests would wrongly need 'use re
    # "eval"', or would assert fail, or crash, or produce unpredictable
    # results.

    local $" = '-'; # separator when interpolating arrays

    my $pat;

    my $A = 'A';
    my $B = 'B';
    my $C = 'C';
    my $D = 'D';
    my $E = 'E';

    my $a = 'aa';
    my $b = 'bb';
    my $c = 'cc';
    my $d = 'dd';
    my $e = 'ee';

    my @r = (qr/(??{$B})/);

    # array with single element, usually following a literal code block

    like "B",   qr/^@r$/,                   "code in array 1";
    like "AB" , qr/^(??{$A})@r$/,           "code in array 2";
    like "XAB", qr/^X(??{$A})@r$/,          "code in array 3";
    $pat =   qr/^X(??{$A})@r(??{$C})$/;
    like "XABC",    $pat,                    "code in array 4";
    unlike "",      $pat,                    "code in array 4 not 1";
    unlike "XAC",   $pat,                    "code in array 4 not 2";
    unlike "XAbbC", $pat,                    "code in array 4 not 3";

    {
        my $B = 'Q';
        push  @r, qr/(??{$B})/;
    }

    # array with two elements, usually following a literal code block
    #
    like "B-Q",    qr/^@r$/,                  "code in array 5";
    like "AB-Q",   qr/^(??{$A})@r$/,          "code in array 6";
    like "XAB-Q",  qr/^X(??{$A})@r$/,         "code in array 7";
    $pat =   qr/^X(??{$A})@r(??{$C})$/;
    like "XAB-QC",   $pat,                    "code in array 8";
    unlike "",       $pat,                    "code in array 8 not 1";
    unlike "XAC",    $pat,                    "code in array 8 not 2";
    unlike "XAB-BC", $pat,                    "code in array 8 not 3";

    # Simple overload package which returns a lower-cased version
    # of a concatenated string, with a '=' used to join

    package LcConcat {
        use overload
            '""' => sub { ${$_[0]} },
            '.' =>  sub {
                            my ($x, $y) = @_[ $_[2] ? (1,0) : (0,1) ];
                            my ($xx, $yy) = ("$x", "$y");
                            lc("$xx=$yy");
                        }
            ;
    }

    my $r = qr/(??{$E})/;
    bless $r, 'LcConcat';

    # Overloading concatenation converts literal compile-time code
    # blocks into run-time recompiled affairs, so need to enable eval
    use re 'eval';

    # First, use an overloaded *scalar* to establish baseline
    # behaviour (i.e. not yet using an array of scalars).
    # Note that the overloaded concatenation converts everything in
    # the pattern to its left to lowercase, so (??{$B}) becomes
    # (??{$b}) etc.

    like "=ee",     qr/^$r$/,                  "code in array 9";
    {
        no re 'eval';
        eval q{my $x = qr/^$r$/; 1};
        like $@, qr/Eval-group not allowed/,   "code in array 9 - err";
    }
    like "aa=ee",   qr/^(??{$A})$r$/,          "code in array 10";
    like "xaa=ee",  qr/^X(??{$A})$r$/,         "code in array 11";
    $pat = qr/^X(??{$A})$r(??{$C})$/;
    like "xaa=eeC",  $pat,                     "code in array 12";
    unlike "",       $pat,                     "code in array 12 not 1";
    unlike "XA=EC",  $pat,                     "code in array 12 not 2";

    # Then add an overloaded scalar to an *array* to see if it's
    # still handled ok by the array interpolation code

    push @r, $r;

    like "bb-bb-=ee",     qr/^@r$/,            "code in array 13";
    {
        no re 'eval';
        eval q{my $x = qr/^@r$/; 1};
        like $@, qr/Eval-group not allowed/,   "code in array 13 - err";
    }
    like "aabb-bb-=ee",   qr/^(??{$A})@r$/,    "code in array 14";
    like "xaabb-bb-=ee",  qr/^X(??{$A})@r$/,   "code in array 15";
    $pat = qr/^X(??{$A})@r(??{$C})$/;
    like "xaabb-bb-=eeC",  $pat,               "code in array 16";
    unlike "",             $pat,               "code in array 16 not 1";
    unlike "XAB-B-=EC",    $pat,               "code in array 16 not 2";

    1;
};
