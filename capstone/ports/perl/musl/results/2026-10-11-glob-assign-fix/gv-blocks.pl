# op/gv.t (5.36.3) lines 1139-1155 (RT #125840) and 960-976 (perl #121242), with is()/pass()
# replaced by prints so they run without t/test.pl.
{
    $a_125840 = 1;
    $b_125840 = 2;
    $a_125840 = *b_125840;
    *a_125840 = $a_125840;
    print "RT125840 *a = \$a: got $a_125840 want 2\n";
    $c_125840 = 1;
    $d_125840 = 2;
    *d_125840 = $d_125840 = *c_125840;
    print "RT125840 *d=\$d=*c: got $d_125840 want 1\n";
    $c_125840 = $d_125840;
    print "RT125840 \$c=\$d: got $c_125840 want 1\n";
}
sub Fred::AUTOLOAD { $Fred::AUTOLOAD }
undef *{"Fred::AUTOLOAD"};
print "121242 no crash from gp_free triggering gv_try_downgrade\n";
sub _121242::DESTROY { delete $_121242::{$_[0][0]} };
${"_121242::foo"} = bless ["foo"], _121242::;
undef *{"_121242::foo"};
print "121242 no crash from pp_undef/gp_free freeing the gv\n";
${"_121242::bar"} = bless ["bar"], _121242::;
*{"_121242::bar"} = "bar";
print "121242 no crash from sv_setsv/gp_free freeing the gv\n";
${"_121242::baz"} = bless ["baz"], _121242::;
*{"_121242::baz"} = *foo;
print "121242 no crash from glob_assign_glob/gp_free freeing the gv\n";
print "GV_BLOCKS_DONE\n";
