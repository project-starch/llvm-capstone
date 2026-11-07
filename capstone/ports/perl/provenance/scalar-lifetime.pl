#!/usr/bin/perl
# Independent regression for retained, freed SV arena heads. Perl 5.22 rejects
# these with its ordinary caught exception; metadata lookup must not read the
# former scalar body first. No benchmark test or input is included here.
use strict;
use warnings;
$| = 1;
my ($checks, $failures) = (0, 0);
sub check {
    my ($yes, $name) = @_;
    ++$checks;
    ++$failures unless $yes;
    print(($yes ? 'ok ' : 'not ok '), "$checks - $name\n");
}
my $target = 'lifetime-target';
for my $kind ('integer', 'string', 'packed') {
    my @values = map {
        $kind eq 'integer' ? $_ :
        $kind eq 'string' ? "value-$_" : pack('p>', $target)
    } 1..9;
    my $finished = eval {
        my @result = sort { undef @values; 1 } (@values, 11);
        1;
    };
    my $error = $@;
    check(!$finished && $error =~ /panic: attempt to copy freed scalar/,
          "$kind freed element reaches core exception");
    @values = (7, 2, 5);
    my @normal = sort { $a <=> $b } @values;
    check(join(',', @normal) eq '2,5,7', "$kind sort works after exception");
}
my $packed = pack('p>', $target);
my @copies = sort { 0 } ($packed, $packed);
check(unpack('p>', $copies[0]) eq $target,
      'live sorted packed value retains authority');
check(unpack('p>', $copies[1]) eq $target,
      'live sorted packed sibling retains authority');
print "1..$checks\n";
exit($failures ? 1 : 0);
