use strict;
use warnings;
my ($n, $batches, $retained) = @ARGV;
sub phase { my $s = "MEMPHASE $_[0]\n"; syswrite(STDERR, $s); }
sub records {
    my ($count) = @_;
    my @out;
    for my $i (0 .. $count - 1) {
        my $line = "$i,alpha:beta," . ("x" x 96);
        my ($id, $tags, $body) = split /,/, $line;
        push @out, { id => 0 + $id, tags => [split /:/, $tags], body => $body };
    }
    return \@out;
}
my $keep = records($retained);
phase('baseline');
my $checksum = 0;
for my $epoch (0 .. $batches - 1) {
    my $batch = records($n * ($epoch == int($batches / 2) ? 4 : 1));
    $checksum += $_->{id} for @$batch;
    phase("live-$epoch");
    undef $batch;
    phase("released-$epoch");
}
my $sum = 0;
$sum += $_->{id} for @$keep;
die "retained oracle" unless $sum == $retained * ($retained - 1) / 2;
my $burst = 4 * $n;
die "batch oracle" unless $checksum == ($batches - 1)*$n*($n-1)/2 + $burst*($burst-1)/2;
print "EXP-OK perl $checksum\n";
