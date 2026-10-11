# row16_datasette_authorizer_ctx

Upstream: https://github.com/datasette/datasette-sqlite-authorizer/issues/3
Table 3 row: 16
Class: UAF
Essence: An authorizer callback remains registered after its heap context is freed and dereferences that context during statement preparation.
Observed: ERROR: AddressSanitizer: heap-use-after-free
