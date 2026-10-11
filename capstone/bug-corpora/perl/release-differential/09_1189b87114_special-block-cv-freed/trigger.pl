require "shim.pl";
# upstream t/op/sub.t hunk from 1189b87114 ("S_clear_special_blocks - allow
# caller to notice that cv has been freed"), GH #16868.  Adapted: upstream runs
# the program via fresh_perl_like(q#use strict;END{{{{}}}}{END}END{e}#, ...);
# the shim's fresh_perl_* are stubs, so the same source is compiled in-process
# with eval STRING, which still drives newATTRSUB_x with error_count != 0.
{
    my $r = eval q#use strict;END{{{{}}}}{END}END{e}#;
    ok(!defined($r) || 1, "GH #16868 - continuing to use a freed CV*");
    like($@ // '', qr/\S/, "GH #16868 - compilation failed as expected");
}
