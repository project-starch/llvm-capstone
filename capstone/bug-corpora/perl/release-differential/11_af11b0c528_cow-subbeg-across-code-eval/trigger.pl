require "shim.pl";
# upstream t/re/pat_re_eval.t hunk from af11b0c528 ("regexec.c - preserve
# COW-backed subbeg state across /(?{})/"), GH #16952 / RT #134026 / GH #24338.
# Adapted: the three upstream cases are fresh_perl_is programs; the shim's
# fresh_perl_is is a stub, so each program body runs in-process and its output
# is accumulated into a string and compared.
use re 'eval';

{
    my $out = '';
    for ("foo", "bar") {
        /f(o)o|(?{})baz/;
        $out .= defined($&) ? "$&" : ''; $out .= '-';
        $out .= defined($1) ? "$1" : ''; $out .= "\n";
    }
    is($out, "foo-o\nfoo-o\n", '[perl #16952] failed (?{}) branch keeps prior captures');
}

{
    my $out = '';
    my ($good, $bad) = qw(ab c);
    for ($good, $bad) {
        /b|(?{})d/;
        $out .= defined($&) ? $& : '';
    }
    is($out, "bb", '[perl #16952] failed (?{}) branch does not assert fetching $&');
}

{
    my $out = '';
    my ($good, $bad) = qw(ab cd);
    for ($good, $bad) {
        s/ b | (?{ 1; }) e //x;
        $out .= "$_: " . (defined($&) ? $& : '') . "\n";
    }
    is($out, "a: b\ncd: b\n", '[perl #16952] substitution form does not assert fetching $&');
}
