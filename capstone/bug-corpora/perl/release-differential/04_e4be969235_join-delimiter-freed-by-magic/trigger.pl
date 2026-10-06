require "shim.pl";
# upstream t/op/join.t hunks from e4be969235 ("join: save the delimiter string
# before anything magical happens to it"), GH #21458 / GH #21484.
# The commit un-TODOs the self-modifying-tie tests and two fresh_perl_is
# programs.  Adapted: the two fresh_perl_is program bodies are run in-process
# (the shim's fresh_perl_is is a stub) and their "print" replaced by an is().

# self-modifying tied variable (context sub from the same file)
{
    package SM;
    our $fetched;
    sub TIESCALAR { my $x = "1";   $fetched = 0; bless \$x }
    sub FETCH     { my $y = shift; $fetched++;   $$y += 3 }

    package main;
    my $t;

    tie $t, "SM";
    is( join( $t, 'a' ), 'a', 'tied separator on single item join' );
    is( $SM::fetched,    0,   'FETCH not called' );

    tie $t, "SM";
    is( join( $t, "a", $t, "b", $t, "c" ),
        'a474b4104c', 'tied separator also in the join arguments' );
    is( $SM::fetched, 3, 'FETCH called 1 + 2 times' );
}
{
    # see GH #21484 -- "modifications delim from magic should be ignored"
    my $n = 1;
    my $sep = "\x{100}" x $n;
    package MyOver1 {
        use overload '""' => sub { $sep = "\xFF" x $n; "x" };
    }
    my $x = bless {}, "MyOver1";
    is( join($sep, "a", $x, "b"), "a\x{100}x\x{100}b",
        "modifications delim from magic should be ignored" );
}
{
    # see GH #21484 -- "modifications to delim PVX shouldn't crash"
    my $n = 1;
    my $sep = "\x{100}" x $n;
    package MyOver2 {
      use overload '""' => sub { $sep = "\xFF" x ($n+20); "x" };
    }
    my $x = bless {}, "MyOver2";
    is( join($sep, $x, "a"), "x\x{100}a",
        "modifications to delim PVX shouldn't crash" );
}
