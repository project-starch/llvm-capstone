<?php
// Minimized trigger for streams.c:350 heap-use-after-free (php-5.0.0 ASan).
// Root cause: registering a userspace stream wrapper makes a REQUEST-LOCAL
// ("volatile") heap copy of the URL-wrapper hash. A stream left open at
// shutdown keeps a raw pointer into that hash. On request teardown the
// volatile wrapper hash is destroyed by module deactivation BEFORE the
// resource list is torn down, so the stream dtor dereferences a dangling
// stream->wrapper (READ of stream->wrapper->wops at streams.c:350).
class test {}
stream_wrapper_register('test', 'test');
$sfp = fopen("php://stdout", "w");   // never closed -> destroyed at shutdown
