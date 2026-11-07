use strict;
use warnings;
my $mode = shift || 'catch';
my $source = 'kept live throughout the fault';
my $second = 'unaffected field';
my $width = length(pack('p', $source));
if ($mode eq 'oom') {
    my $packed = pack('p', $source);
    fail_replace($packed, $source, 0, 2);
    die "OOM unexpectedly returned\n";
}
my $checks = 0;
for my $case (0 .. 31) {
    my $packed = pack('pp', $second, $source);
    eval { fail_replace($packed, $source, $width, 1) };
    die "missing expected exception: $@\n" unless $@ =~ /test-only allocation boundary exception/;
    $checks++;
    die "surviving authority lost\n" unless unpack('p', $packed) eq $second;
    $checks++;
    eval { my $discard = unpack('x' . $width . 'p', $packed) };
    die "replaced authority survived fault\n" unless $@ =~ /without matching provenance/;
    $checks++;
    undef $packed;
    $checks++;
}
print "ALLOCATION_CHECKS checks=$checks failures=0\n";
