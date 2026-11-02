<?php
// unregister_tick_function() inside the tick function: zend_llist_apply is mid-iteration
// over user_tick_functions; the callback deletes (efree) the current element, then the
// apply loop dereferences element->next on the freed node -> use-after-free.
function a() {
    unregister_tick_function('a');
}
declare(ticks=1);
register_tick_function('a');
$x = 1;
$y = 2;
$z = $x + $y;
echo "done\n";
