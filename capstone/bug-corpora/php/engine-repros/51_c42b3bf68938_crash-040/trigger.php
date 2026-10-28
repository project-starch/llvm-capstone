<?php
class Evil { public $r; function __destruct() { unset($GLOBALS['victim']); $GLOBALS['n']=new Evil(); } }
class Trig { function __destruct() { $GLOBALS['victim']=new Evil(); for($i=0;$i<40;$i++)$GLOBALS['z'.$i]=new Evil(); } }
$t=new Trig();
