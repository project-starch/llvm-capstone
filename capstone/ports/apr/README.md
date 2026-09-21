# APR

Two things live here, and they are different in kind.

| | |
|---|---|
| [`census-apr.sh`](census-apr.sh), [`build-apr-census.sh`](build-apr-census.sh), [`fetch-apr.sh`](fetch-apr.sh) | the **census**: what APR's pool allocator would cost to bring under the discipline, counted from the source, and whether `apr_pools.c` compiles freestanding at all. No domain, no run |
| [`pools/`](pools/README.md) | the **port** that grew out of it: the same allocator in a Capstone domain with a Sublet adapter, for the [httpd/APR bug corpus](../../bug-corpora/httpd/apr-pool-repros/README.md) |
| [`adapted/apr_shim.h`](adapted/apr_shim.h) | the one header both use in place of the fourteen APR includes |

The census keeps upstream's numbers; the port's one ABI change, 16-byte
default alignment, is a knob the shim defaults off.
