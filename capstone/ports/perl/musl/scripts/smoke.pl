# What a domain must get right before anything else is measured on it. Core
# builtins only: the interpreter's own library is not on the share, so a `use`
# would fail for a reason that has nothing to do with the port.
my $dir = $ARGV[0] || "/mnt/host/files";
print "P1 hello\n";
my @a = map { $_ * 2 } (1 .. 2000);
printf "P2 array %d %d\n", scalar(@a), eval { my $s = 0; $s += $_ for @a; $s };
my %h; $h{"k$_"} = $_ for (0 .. 2999);
print "P3 hash ", scalar(keys %h), " $h{k2999}\n";
my $s = ""; $s .= "ab$_" for (0 .. 499);
print "P4 string ", length($s), " ", substr($s, 0, 10), "\n";
# SV churn: the arenas hand out and reclaim heads and bodies for every one of
# these, which is the layer this port exists to put under Sublet.
my @objs; push @objs, { i => $_, s => "s$_", a => [$_, $_ + 1] } for (0 .. 19999);
@objs = (); print "P5 churn ok\n";
sub fib { my $n = shift; $n < 2 ? $n : fib($n - 1) + fib($n - 2) }
print "P6 fib ", fib(20), "\n";
sub deep { my $n = shift; $n == 0 ? 0 : 1 + deep($n - 1) }
print "P7 deep ", deep(500), "\n";
print "P8 eval ", (eval { die "x\n"; 1 } ? "no" : "caught"), " ", ($@ =~ /^x/ ? "msg" : "nomsg"), "\n";
printf "P9 num %.2f %s %s\n", 1.5 * 3, 10 / 4, 2**70;
my @w = sort { $a cmp $b } qw(pear apple fig banana);
print "P10 sort @w\n";
my $t = "The quick brown fox";
my @m = ($t =~ /(\w+)\s+(\w+)/);
(my $u = $t) =~ s/quick/slow/;
print "P11 regex @m | $u | ", scalar(() = $t =~ /o/g), "\n";
my $cl = do { my $c = 0; sub { ++$c } };
$cl->() for (1 .. 4);
print "P12 closure ", $cl->(), "\n";
my $r = \@a; my $rr = \$r;
print "P13 ref ", ref($r), " ", ref($rr), " ", $$rr->[0], "\n";
# A blessed object and a method call: this is the path that faulted first, since
# a method call asks for the package's stash (patches 0003 and 0004).
{ package Counter; sub new { bless { n => 0 }, shift } sub inc { $_[0]{n}++; $_[0] } sub n { $_[0]{n} } }
my $o = Counter->new; $o->inc->inc->inc;
print "P14 method ", ref($o), " ", $o->n, "\n";
open(my $fh, ">", "$dir/out.txt") or die "open: $!";
print $fh "line $_\n" for (0 .. 49);
close $fh;
open($fh, "<", "$dir/out.txt") or die "reopen: $!";
my @lines = <$fh>; close $fh;
print "P15 file ", scalar(@lines), " ", ($lines[49] =~ /49/ ? "last-ok" : "last-bad"), "\n";
opendir(my $dh, $dir) or die "opendir: $!";
my @ents = grep { /smoke\.pl/ } readdir($dh); closedir $dh;
print "P16 dir ", scalar(@ents), "\n";
print "P17 pack ", join(",", unpack("C*", pack("C*", 1, 2, 3))), " ", sprintf("%05.2f", 3.14159), "\n";
print "SMOKE_DONE\n";
