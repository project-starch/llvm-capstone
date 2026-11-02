<?php
// bug #47353 : object store reallocated inside a destructor leaves the
// del_ref `obj` pointer dangling -> use-after-free on obj->refcount.
class Evil {
    public $store = array();
    function __destruct() {
        // Create > 1024 simultaneously-live objects so the object store
        // (initial size 1024) erealloc()s and moves, invalidating the
        // `obj` pointer captured before the dtor call in del_ref.
        for ($i = 0; $i < 4000; $i++) {
            $this->store[$i] = new stdClass();
        }
    }
}
$x = new Evil();
unset($x);   // refcount 1 -> zend_objects_store_del_ref -> dtor -> realloc
echo "ok\n";
