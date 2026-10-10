require "shim.pl";
# upstream t/re/pat_re_eval.t hunk from 39b4841b25 ("Fix crash on recursive
# /(?{...})/ call"), GH #22869.  Verbatim.
use re 'eval';

# GH #22869 "Perl crash with recursive sub and regex with code eval".
#
# A recursive call to a match op with a run-time pattern and which
# contained a code block, led to to the temporary rex stored in the
# OP_MATCH and PL_reg_curpm ops getting prematurely freed when updated
# within the inner match's OP_MATCH op.

{
    my @got;

    my $f = sub {
        my ($s, $re) = @_;
        $s =~ $re;
        push @got, ',', $1, $2, ']';
    };

    my $pat;
    $pat = qr{^
                (.)
                (?{
                    push @got, '[', $1, $2;
                    $f->('XY', $pat) if $1 eq 'A';
                    push @got, ',', $1, $2;
                })
                (.)
                (?{
                    push @got, ',', $1, $2;
                })
                $
            }x;

    $f->('AB',$pat);

    my $got = join '', map defined ? $_ : '-', @got;
    is($got, "[A-[X-,X-,XY,XY],A-,AB,AB]", "GH22869");
}
