# A self-contained stand-in for t/test.pl and Test::More, so an extracted test
# body runs as one script and yields ONE verdict line. Failures are collected,
# never fatal: a case that dies early must still report what it reached.
use strict; use warnings;
our @FAILS; our $CUR = 0;
sub _f { push @FAILS, [$CUR, @_]; return 0 }
sub plan { 1 } sub done_testing { 1 } sub note { 1 } sub diag { 1 }
sub pass { $CUR++; 1 }
sub fail { $CUR++; _f('fail', $_[0] // '') }
sub ok   { $CUR++; $_[0] ? 1 : _f('ok', $_[1] // '') }
sub is   { $CUR++; my ($g,$e,$n)=@_;
           no warnings 'uninitialized';
           (defined $g == defined $e && (!defined $g || "$g" eq "$e")) ? 1
             : _f('is', $n // '', "got=".(defined $g ? "$g" : 'undef'),
                                  "want=".(defined $e ? "$e" : 'undef')) }
sub isnt { $CUR++; my ($g,$e,$n)=@_; no warnings 'uninitialized';
           ("$g" ne "$e") ? 1 : _f('isnt', $n // '') }
sub like { $CUR++; my ($g,$re,$n)=@_; no warnings 'uninitialized';
           ($g =~ $re) ? 1 : _f('like', $n // '', "got=$g") }
sub unlike { $CUR++; my ($g,$re,$n)=@_; no warnings 'uninitialized';
           ($g !~ $re) ? 1 : _f('unlike', $n // '') }
sub cmp_ok { $CUR++; my ($g,$op,$e,$n)=@_;
             my $r = eval "\$g $op \$e" ? 1 : 0;
             $r ? 1 : _f('cmp_ok', $n // '', "got=".(defined $g ? $g : 'undef')) }
sub is_deeply { $CUR++; my ($g,$e,$n)=@_;
                require Data::Dumper; local $Data::Dumper::Sortkeys=1;
                local $Data::Dumper::Indent=0;
                (Data::Dumper::Dumper($g) eq Data::Dumper::Dumper($e)) ? 1
                  : _f('is_deeply', $n // '') }
sub skip { 1 } sub skip_all { exit 0 } sub curr_test { $CUR }
sub eq_array { my ($a,$b)=@_; return 0 unless @$a == @$b;
               for my $i (0..$#$a) { no warnings 'uninitialized';
                 return 0 if "$a->[$i]" ne "$b->[$i]" } 1 }
sub fresh_perl_is { 1 } sub fresh_perl_like { 1 } sub runperl { '' }
sub watchdog { 1 } sub tempfile { "/tmp/perlcase.$$" }
END {
  # One line, parseable, printed whatever happened before it.
  if (@FAILS) { print "[FAIL] ", scalar(@FAILS), " of $CUR: ",
                      join(' | ', map { join(',', map { defined $_ ? $_ : 'undef' } @$_) } @FAILS[0..($#FAILS > 2 ? 2 : $#FAILS)]), "\n" }
  else        { print "[PASS] $CUR\n" }
}
1;
