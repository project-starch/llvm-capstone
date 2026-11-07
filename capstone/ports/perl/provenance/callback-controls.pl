#!/usr/bin/perl
# Independent adapter review probes. No benchmark inputs or tests are changed.
# Run --positive-only under an ordinary Perl; default additionally requires the
# provenance adapter's controlled rejection of raw byte reconstruction.
$| = 1;
my ($checks, $failures) = (0, 0);
my $negative = !@ARGV || $ARGV[0] ne '--positive-only';
sub check {
    my ($yes, $name) = @_;
    ++$checks;
    ++$failures unless $yes;
    print(($yes ? 'ok ' : 'not ok '), "$checks - $name\n");
}
sub pointer_ok {
    my ($value, $name, $format) = @_;
    $format = 'p>' unless defined $format;
    my $result = eval { unpack($format, $value) };
    my $error = $@;
    check(!$error && defined($result) && $result eq 'review-target', $name);
    print "# exception: $error" if $error;
}
sub pointer_rejected {
    my ($value, $name, $format) = @_;
    return unless $negative;
    $format = 'p>' unless defined $format;
    my $ok = eval { unpack($format, $value); 1 };
    check(!$ok && $@ =~ /provenance/, $name);
}
my $target = 'review-target';
my $packed = pack('p>', $target);
my $width = length($packed);
my $raw = pack('H*', unpack('H*', $packed));
pointer_ok($packed, 'big-endian positive control');
pointer_rejected($raw, 'fresh raw bytes carry no authority');

# Pure observations must not revoke the source's authority.
for my $operation (
    ['length', sub { my $n = length($_[0]); }],
    ['equality', sub { my $b = $_[0] eq $_[0]; }],
    ['cmp', sub { my $b = $_[0] cmp $_[0]; }],
    ['regex', sub { my $b = $_[0] =~ /./s; }],
    ['regex pos cache', sub { my $b = $_[0] =~ /./sg; pos($_[0]) = 0; }],
    ['index', sub { my $i = index($_[0], ''); }],
    ['rindex', sub { my $i = rindex($_[0], ''); }],
    ['ord', sub { my $n = ord($_[0]); }],
    ['hex observation', sub { my $h = unpack('H*', $_[0]); }],
) {
    my $value = $packed;
    $operation->[1]->($value);
    pointer_ok($value, 'read preserves: ' . $operation->[0]);
    utf8::upgrade($value);
    $operation->[1]->($value);
    pointer_ok($value, 'UTF8 read preserves: ' . $operation->[0]);
}
my $reused = chr(256) x 80;
my $cache_len = length($reused);
my $cache_slice = substr($reused, 30, 10);
$reused =~ /./g;
$reused = $packed;
utf8::upgrade($reused);
pointer_ok($reused, 'reused UTF8 and regex cache scalar');

my $suffix = $packed;
$suffix .= 'x';
pointer_ok($suffix, 'in-place suffix preserves prefix');
$suffix = $packed;
$suffix .= chr(256);
pointer_ok($suffix, 'in-place UTF8 suffix preserves prefix');
my $prefix = chr(256) . $packed;
pointer_ok($prefix, 'UTF8 prefix and byte pointer concat', 'xp>');
my $utf = $packed;
utf8::upgrade($utf);
my $copied = unpack('a*', $utf);
pointer_ok($copied, 'unpack a from UTF8 pointer');
my $replaced = 'x' . $packed;
substr($replaced, 0, 1, chr(256));
pointer_ok($replaced, 'UTF8 replacement preserves trailing field', 'xp>');
my $inserted = chr(256) . 'x';
substr($inserted, 1, 1) = $packed;
pointer_ok($inserted, 'UTF8 substring insertion after multibyte character', 'xp>');

# Real lvalue aliases exercise get/set magic, not optimized substr assignment.
my $destination = 'ab';
my $alias = \substr($destination, 1, 0);
$$alias = $packed;
pointer_ok($destination, 'substring lvalue alias insertion', 'xp>');
my $whole_alias = \substr($destination, 1, $width);
my $from_alias = $$whole_alias;
pointer_ok($from_alias, 'substring lvalue alias fetch');
my $partial_alias = \substr($destination, 1, 1);
$$partial_alias = substr($raw, 0, 1);
pointer_rejected(substr($destination, 1, $width), 'partial alias same-byte overwrite');

{
    package ReviewBox;
    sub TIESCALAR { bless { value => $_[1], fetch => 0, store => 0 }, $_[0] }
    sub FETCH { ++$_[0]->{fetch}; return $_[0]->{value}; }
    sub STORE { ++$_[0]->{store}; $_[0]->{value} = $_[1]; }
}
my $box = tie my $tied, 'ReviewBox', 'alpha';
my $normal = $tied;
check($normal eq 'alpha' && $box->{fetch} == 1, 'assignment FETCH exactly once');
$box->{fetch} = 0;
$normal = 'x' . $tied;
check($normal eq 'xalpha' && $box->{fetch} == 1, 'concat FETCH exactly once');
$box->{fetch} = 0;
$normal = join('', 'x', $tied, 'y');
check($normal eq 'xalphay' && $box->{fetch} == 1, 'join FETCH exactly once');
$tied = 'beta';
check($box->{store} == 1 && $box->{value} eq 'beta', 'assignment STORE exactly once');
$box->{value} = $packed;
$box->{fetch} = 0;
$normal = $tied;
check($box->{fetch} == 1, 'packed value FETCH exactly once');
pointer_ok($normal, 'explicit packed value returned by FETCH');
$box->{fetch} = 0;
$normal = $tied . 'x';
check($box->{fetch} == 1, 'packed left concat FETCH exactly once');
pointer_ok($normal, 'direct packed FETCH as concat left');
$box->{fetch} = 0;
$normal = 'x' . $tied;
check($box->{fetch} == 1, 'packed right concat FETCH exactly once');
pointer_ok($normal, 'direct packed FETCH as concat right', 'xp>');
$box->{fetch} = 0;
$normal = substr($tied, 0, $width);
check($box->{fetch} == 1, 'packed substring FETCH exactly once');
pointer_ok($normal, 'direct packed FETCH as substring source');
$box->{value} = $raw;
$box->{fetch} = 0;
$normal = $tied;
check($box->{fetch} == 1, 'raw replacement FETCH exactly once');
pointer_rejected($normal, 'raw FETCH cannot revive previous packed value');

# All negative controls preserve the final bytes, so memcmp alone cannot pass.
my $mutated = $packed;
vec($mutated, 0, 8) = vec($mutated, 0, 8);
pointer_rejected($mutated, 'same-byte vec write');
$mutated = $packed;
$mutated =~ s/(.)/$1/s;
pointer_rejected($mutated, 'same-byte substitution');
$mutated = $packed;
$mutated =~ s/(.)/$1/se;
pointer_rejected($mutated, 'same-byte evaluated substitution');
my $head = substr($packed, 0, 1);
my $tail = substr($packed, 1);
pointer_rejected($head . $tail, 'partial fragments cannot reconstruct a field');
pointer_rejected(pack('p>@0a*', $target, $raw), 'pack absolute rewind and raw restoration');
$mutated = $packed;
utf8::encode($mutated);
utf8::decode($mutated);
pointer_rejected($mutated, 'byte encoding boundary does not restore authority');
my $sibling = $packed;
$mutated = $packed;
substr($mutated, 0, $width) = $raw;
pointer_rejected($mutated, 'complete raw overwrite is not provenance');
pointer_ok($sibling, 'sibling survives independent raw overwrite');
if ($negative) {
    for my $format ('p', 'p<', 'p>') {
        my $original = pack($format, $target);
        open my $fh, '>', 'review-pointer-io.tmp' or die $!;
        binmode $fh;
        print {$fh} $original;
        close $fh or die $!;
        my $readback = $original;
        open $fh, '<', 'review-pointer-io.tmp' or die $!;
        binmode $fh;
        my $got = read($fh, $readback, $width);
        close $fh or die $!;
        unlink 'review-pointer-io.tmp' or die $!;
        check(defined($got) && $got == $width, "$format full file read control");
        pointer_rejected($readback, "$format file read destroys sidecar and native tag", $format);
        pointer_ok($original, "$format file write leaves source authority", $format);
    }
}
for (1..20) {
    my $copy = $packed;
    eval { $copy =~ s/(.)/die "review-throw\n"/se; };
    pointer_rejected($copy, 'exception after mutation entry invalidates provenance');
    pointer_ok($packed, 'exception cleanup preserves independent source');
}
print "1..$checks\n";
print "REVIEW_CALLBACKS:checks=$checks:failures=$failures\n";
exit($failures ? 1 : 0);
