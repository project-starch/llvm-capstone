# tshark as a full application in a Capstone domain — M0 census and port plan (2026-09-23)

**Status:** a plan with its M0 census, not the port. It lives on branch `tshark-app`, like the
FFmpeg app port it follows (`2026-09-23-ffmpeg-full-app-port.md`). An adversarial audit ran
before implementation; its corrections are folded in, and the three claims it refuted are listed
under "Withdrawn" so the trail survives.

**Question:** can tshark (Wireshark 4.6.8) read a capture file and print its full dissection
inside one Capstone domain, and what does it cost? In particular, what does it cost with the
heap **bounded per object and revoked on free**? The FFmpeg port showed that correctness under
enforcement and safety are different results
(`ports/ffmpeg/app/results/2026-09-23-qemu-safety/`), so safety is part of acceptance here
from the start.

**Scope:** QEMU only. Board work is a later, separate question: this ABI depends on QEMU's
fabricated `gp`, and the deployed silicon lets a stale data access retire (ISSUES Q-11).

**Every number below comes from a run on 2026-09-23** unless it is marked "computed". The
census builds are throwaway and live under `/tmp/capstone/tshark-app/`. What the census could
not settle is marked **UNRESOLVED**.

## The answer in one paragraph

A minimal tshark works natively. It has 19 whitelisted dissectors, 9 stubs, and 3 empty
tables, and its full `-V` dissection of four shipped captures (10 frames) is byte-identical to
stock tshark's. The comparison can fail: it DIFFERS on six other shipped captures whose
protocols the whitelist leaves out.

Porting it into a domain is a different order of work from FFmpeg:
- **Image and heap:** the Wireshark libraries alone are 16.2 MiB natively. The dependencies add
  about 5.3 MiB. The heap would reserve about **45 MiB during `epan_init`** on capstone64
  (computed) unless one preallocation constant is scaled down. So the domain block is **64–128
  MiB**, from CMA, until the heap is measured.
- **Libraries:** GLib, libgcrypt (with libgpg-error), c-ares, PCRE2 and **libxml2** are all
  required by 4.6.8, plus zlib.
- **Compile blockers found:** two. `CF_FUNC` is fixed in one line. Pointer-typed `__thread`
  (ISSUES C-47) is hit in 3 translation units, including the exception stack every dissector
  uses.
- **GLib:** 20 of its 96 core files do not compile for capstone64. Ten of them trip GLib's own
  assertions that a pointer is word-sized. Its once-init macro casts pointers through
  `guintptr`.
- **Provenance through GLib's integer macros is UNMEASURED.** The compiler warning used for the
  census cannot see it (below).

## Withdrawn (reported earlier in the session, refuted by the audit, verified)

- **"Production code: 0 flagged pointer round trips."**
  - `-Wcapstone-pointer-roundtrip` recognises only a value spelled `uintptr_t`/`intptr_t` and a
    single cast (`clang/lib/Sema/SemaCapstone.cpp:26-48`).
  - It is silent on `GSIZE_TO_POINTER(GPOINTER_TO_SIZE(p))` and even on a plain `gsize v =
    (gsize)p; return (void *)v;`. Verified: 0 flags on both, in a direct test with the census
    flags.
  - The positive control used a `uintptr_t` typedef. So it proved that the warning fires, not
    that it sees the class being measured.
  - The linked sources hold 79 `GPOINTER_TO_UINT`, 19 `GPOINTER_TO_INT`, 170 `GUINT_TO_POINTER`
    and 55 `GINT_TO_POINTER` uses. The round trips among them are **unmeasured**, and the audit
    found one real instance the census missed (`ui/voip_calls.c:145`; not linked into tshark).
- **"One compile-blocking macro."** A pointer-typed `__thread` fails in codegen: `Cannot
  select: GlobalTLSAddress` (C-47). Verified on a sample. A `-fsyntax-only` census cannot see a
  codegen failure.
- **"A domain block of about 32 MiB."** It ignored the heap, and the dependencies.

## M0 census

### Source and workload

- **Source:** `wireshark-4.6.8.tar.xz`, sha256 `c0f1ccf2…`, verified. It is the same pin as
  `ports/wireshark/wmem/upstream.json`; `upstream.json` here is a copy of it.
- **Workload:** four captures shipped in the tarball (`test/captures/`), so they are pinned by
  its hash:
  - `dhcp.pcap`, `dns_port.pcap`, `http.pcap`, `arp.pcap`: 10 frames;
  - protocols: eth, ethertype, ip, udp, tcp, dns, dhcp, http, arp, fr.
- **Oracle:** `TZ=UTC tshark -r <capture> -V -n`, from a **stock** build of the pristine
  tree, with all dissectors (`native-stock/`).
  - It is deterministic: two runs are byte-identical, and stderr is empty.
  - `-n` matters: without it, MAC names are resolved. TZ matters too.
  - **To add:** pin `HOME`/`WIRESHARK_CONFIG_DIR` to an empty directory. The audit reran with an
    empty `HOME` and got the same result, but nothing in the harness pins it yet.
- **Controls:**
  - **Flipped byte:** one byte, 8 from the end of each capture, changes **1** line of stock
    output, and the minimal build still matches stock. It lands in a leaf field of each
    capture's last frame, so it tests input sensitivity, not dissector choice.
  - **Harness, negative (from the audit):** stock and minimal DIFFER on `ntp.pcap`,
    `retrans-tls.pcap`, `tftp.pcap`, `ipx_rip.pcap`, `ipv6.pcap` and `http-ooo.pcap`, and match on
    `dns-ooo.pcap`. The comparison detects a missing dissector, and these stay in the harness as
    its controls.

### A minimal tshark, natively (`dissector-whitelist.txt`, `patches/0001-0003`, `src/capstone-stubs.c`)

- **Build:** a fresh extraction, plus the three patches and the stub file, equals the tested
  tree exactly (`diff -rq`, audit). The two CMake caches differ only in the whitelist.
- **Whitelist:** frame, eth, ethertype, ip, ipv6, udp, tcp, dns, dhcp, http, arp, fr, chdlc,
  ieee8023, isl, llc, xdlc, media-type, data. These are the workload's protocols plus what their
  dissectors reference; four rounds of link errors and crashes found them.
- **Stubs (9)** for couplings to dissectors left out, each output-neutral on the workload:
  - **4 run at every start:** DCE/RPC reset from `decode_as.c:533`; TLS and DTLS port
    registration from the http, dns and data dissectors; and `fcs_options`, an empty enum where
    stock has 3 entries.
  - **5 are never reached.**
  - Pulling TLS in instead would drag in BER, X.509, OCSP, QUIC and SRTP.
- **Empty tables (3)** that whitelisted dissectors look up but whose owners are left out:
  `osinl.incl`, `streaming_content_type` and `llc.hpteam_pid`. An empty table answers "nothing
  registered". `osinl.incl` **is consulted** on the workload: the Frame Relay frame in
  `arp.pcap` carries NLPID 0x80. Stock has no 0x80 entry either, so the answer is the same.
- **Three core NULL dereferences** that only a whitelisted build reaches (patch 0002):
  - `dissector_add_range_preference` reads `ui_name` through a missing table (`packet.c:1397`,
    from `packet-http.c:5206`'s `sctp.port`);
  - the SRT, RTD and stat-tap iterators walk a tree that only a registration creates;
  - the postdissector array is created by the first registration, and 3 of the 7 loops over it
    are unguarded.
- **More absent-owner paths remain (audit), not patched:**
  - handles looked up but never registered: `ipx`/`ccsds` (`packet-ieee8023.c:122,124`),
    `http2`, `tls-echconfig`, `tpkt`;
  - `file` at `packet.c:264-265`, behind a `ws_assert` that the native build disabled with
    `NDEBUG`;
  - the `prefixes` table (`proto.c:1085`), off the `-r` path.

  `ipx_rip.pcap` already reaches one of them: a `DISSECTOR_ASSERT` throws "Dissector bug". **A
  safety fixture must not route into any of these**, or the exception path would read as a
  result.
- **Registrations into absent tables** print 105 `OOPS` lines to stderr, and the core skips
  them.

**Result:** `-V` output is identical to stock on all four captures, and on the flipped inputs.

### Size and heap

| native x86-64, `size` (RelWithDebInfo, no debug) | text | data | bss | total |
|---|---:|---:|---:|---:|
| minimal: libwireshark + libwiretap + libwsutil + tshark | 12.8 MiB | 3.0 MiB | 0.4 MiB | **16.2 MiB** (17.0 MB) |
| its shared-library dependencies (glib, gcrypt, gpg-error, c-ares, pcre2, zlib, libxml2) | | | | about 5.3 MiB |
| stock libwireshark alone | 92 MiB | 36 MiB | 1.9 MiB | 130 MiB |

- **What the native total includes and excludes:**
  - It **includes** 5.8 MiB of `.rela.dyn`, which a static image does not carry. But its 253,969
    relative relocations are pointer initialisers, and each becomes a capability to initialise
    at load. **The cost of initialising them is UNRESOLVED.**
  - `.data.rel.ro` (2.75 MiB) roughly doubles with 16-byte pointers.
- **The heap is not in that table, and it dominates** (from `strace` of a native run):
  - `epan/proto.c:459` sets `PROTO_PRE_ALLOC_HF_FIELDS_MEM` to 305,000, and `proto.c:600`
    reserves that many entries in the field map. That is a 16 MiB `mmap` natively.
  - The field array adds 2.3 MiB, and two 8 MiB wmem blocks come from `guids_init` and
    `addr_resolv_init`.
  - On capstone64 the map items double (3 pointers + a `uint32` becomes 64 B), about 45 MiB
    during `epan_init`. That is **computed**, not measured.
  - The minimal build registers **2,202 fields** (`tshark -G fieldcount`), so the constant can
    be scaled to the whitelist; both structures grow on demand.
- **The module's rule:** without a `.capstone_domreq` declaration, the block is `code_len +
  max(code_len, 64 KiB)`, rounded to a power of two (`capstone.c:152-161`).
- **Estimate:** 64–128 MiB. It becomes a measurement at M0 (the image) and at M2 (the heap after
  `epan_init`).
- **Consulted tables:** the built-in OUI and enterprise tables **are** consulted under `-n`.
  `arp.pcap` prints an OUI organisation. So they stay; only services and similar could go.

### Libraries

| dependency | 4.6.8 | on this host (native) | functions imported by the minimal build |
|---|---|---|---:|
| GLib ≥ 2.54 | required (`CMakeLists.txt:1329`) | 2.80 | 316 (+3 version symbols) |
| libgcrypt ≥ 1.8, plus libgpg-error | required (`:1336`) | missing; built from source (1.11.1 / 1.55) | 43 |
| c-ares ≥ 1.13 | required (`:1339`) | missing; built from source (1.34.5) | 12 |
| PCRE2 | required (`:1346`) | 10.42 | 8 |
| libxml2 ≥ 2.9.7 | **required** (`:1347-1348`); `epan_init` calls `xmlInitParser` (`epan.c:323-324`) | 2.9.14 | used by `epan.c` and 5 wiretap readers |
| zlib | optional (`:1533`) | 1.3 | 13 |

The three hand-fetched dependency tarballs are recorded, **UNVERIFIED** against upstream: the
GnuPG listing was unreachable, and GitHub gives no digest for these assets.
- c-ares-1.34.5 `7d935790…`;
- libgpg-error-1.55 `95b17814…`;
- libgcrypt-1.11.1 `24e91c91…`.

GLib 2.80.5 **is** verified: `9f23a9de…` matches upstream's `.sha256sum`.

**GLib at run time** (audit, `LD_DEBUG=bindings` over the four runs):
- **At most 187 of the 316 are even bound, and 107 are certainly called:** hash tables, trees,
  arrays, lists, GString, GBytes, `g_slice`, `g_regex_new`, `g_utf8_*`, `g_mutex_trylock`.
- **Never called:** thread pools, condition variables, spawn, io_channel, async queues. `strace
  -f` shows 0 `clone` calls.
- `tshark.c:1370` passes `epan_init` no progress callback, so registration runs inline
  (`register.c:65-93`).

So GLib has to be ported, not shimmed, because of the set that is called, not because of 316
link-time imports.

### Pointer provenance and compile blockers

**Wireshark (the minimal build).** The census compiled 384 of the 399 linked sources
`-fsyntax-only` for capstone64. It skipped the 15 generated ones; the audit ran 13 of those
clean.
- **`CF_FUNC`** (`epan/proto.h:91`, `((const void *) (size_t) (x))`) is a compile **error** in
  a static initialiser on capstone64. It is used 2,283 times in 127 dissector files, and **16
  times in the whitelist** (tcp 1, dhcp 13, llc 2). The workload reaches it at run time:
  dhcp's "Renewal Time Value". Patch 0003 casts directly; the native output is unchanged.
- **`__thread`** (`WS_THREAD_LOCAL`, `include/ws_attributes.h:109-113`) fails in codegen
  (C-47) at `epan/except.c:156`, `wiretap/wtap.c:1534` and `wsutil/filesystem.c:2240,2345`. A
  single-threaded domain can define it empty. The fix is still to be made.
- **Codegen pass (audit):** `-S` over all 400 linked sources. 11 fail:
  - 6 on libxml2 headers;
  - 3 on `__thread`;
  - 2 did not finish in 600 s under 48-way load: `epan/manuf.c` with its 4.4 MB table, and
    `pci-ids.c`. **UNRESOLVED.**
- **GLib-macro round trips: UNMEASURED** (see Withdrawn).

**GLib 2.80.5** (`glib/`, `gmodule/`, `gthread/`: 96 files, `-fsyntax-only` for capstone64).
The first run was void: 93 files missed `gversionmacros.h`, which meson generates at build
time. After a native build, **20 of 96 do not compile.**
- **Ten trip GLib's own pointer-is-a-word assertions:**
  - five `g_once_init_enter` sites on a `gsize` (`ggettext.c`, `grand.c`, `gregex.c`,
    `gstrfuncs.c`, `gtimezone.c`). They do pointer-sized atomics on an 8-byte object, which is
    out of bounds under per-object bounds;
  - `gatomic.c:541`;
  - `gdataset.c:1416`;
  - `ghash.c:240` (`GHashTableIter`);
  - `gvariant.c:3227` (`GVariantBuilder`);
  - `glib-init.c:90` (`sizeof (long) == sizeof (void *)`).
- **Ten miss headers:** `linux/futex.h` in 8 files, `linux/wait.h`, and `ALTMON_1`.
- **Round-trip warnings: 3, all in one macro.** `g_once_init_leave_pointer` is
  `(gpointer) (guintptr) (result)` (`glib/gthread.h:292,303`). This is a lower bound, for the
  reason above.
- **Not checked:** GLib's own `__thread`/`G_THREAD_LOCAL` in codegen, and libgcrypt and c-ares
  at all.

### The domain block (M-infra)

- **Guest kernel:** `CONFIG_CMA=y`, `CONFIG_DMA_CMA=y`, `CONFIG_CMA_SIZE_MBYTES=0` (so it needs
  `cma=`), `CONFIG_CMA_ALIGNMENT=8` (1 MiB).
- **The module:** the domain comes from the buddy allocator (`capstone.c:163`,
  `__get_free_pages(order)`, 4 MiB ceiling), while regions already use `dma_alloc_pages`
  (`capstone.c:321`). The module lives in the main checkout's buildroot submodule, not in this
  worktree.
- **The monitor (audit, partly settled):**
  - `split_out_cap` only requires the block to lie inside a live region, with no alignment or
    power-of-two rule;
  - `create_domain` rounds the code/data split to a granule measured from the base, which 1 MiB
    alignment satisfies below 512 MiB;
  - on QEMU the genesis regions cover all guest RAM.
- **UNRESOLVED:** the monitor's 32-bit arithmetic on physical addresses above 4 GiB (it
  documents one such shift itself). Pinning `cma=<size>@<base below 4 GiB>` avoids it.

### Prior art to reuse

- **`ports/wireshark/wmem/`:** Wireshark's wmem allocators in a domain, in `spatial` and `sublet`
  modes, via a guarded hook patch. In a full tshark, wmem is where the temporal story is:
  `block_fast` rewinds the packet pool per packet, invisibly to any system heap.
- **`musl-capstone/runtime/sublet_heap.c`** (on branch `ffmpeg-app`): the system heap bounded per
  object and revoked on free.
- **The FFmpeg port's shape:** staged images that return, a pristine-built oracle, controls,
  pre-registered safety fixtures, and a per-section verdict.

## M0 items still open before implementation (from the audit)

1. A `-S` **codegen** gate over every linked translation unit, run serially, including
   `manuf.c` and `pci-ids.c`.
2. `WS_THREAD_LOCAL` defined empty for the domain. `__thread` checked in GLib, libgcrypt and
   c-ares.
3. A GLib-idiom census that can see nested casts and `gsize`/`guintptr` values, with a positive
   control in the target shape: `GUINT_TO_POINTER(GPOINTER_TO_UINT(p))`. Preprocess and match,
   or extend the Sema check. That extension is the compiler lane's.
4. `PROTO_PRE_ALLOC_HF_FIELDS_MEM` scaled to the whitelist; the block sized from the measured
   native peak heap times the pointer-width growth.
5. The load-time cost of 253,969 capability initialisers.
6. `NDEBUG` pinned, or the `"file"` handle whitelisted or stubbed.
7. The oracle's configuration directories pinned. The harness controls kept: a negative
   (`ntp.pcap` must differ) and a matched pair (`dns-ooo.pcap` must match).
8. The absent-handle paths listed, so that no safety fixture routes into them.

## Plan

Everything happens on `tshark-app`. Nothing lands on `dev` without the lead's OK, and there is no
push without the lead's approval (a new branch).

| milestone | content | exit criterion |
|---|---|---|
| **M0-open** | the eight items above | each settled or recorded as UNRESOLVED with its reason |
| **M-infra** | the domain block from CMA: `capstone.c:163` → `dma_alloc_pages`, `cma=<size>@<base below 4 GiB>` for the QEMU guest, the monitor's rules checked | an existing small domain still runs byte-identically from a CMA block; a 128 MiB block allocates and a domain runs in it. Shared infra, reviewed separately, never folded into the port commit |
| **M-deps** | GLib, libgcrypt/libgpg-error (no asm), c-ares, PCRE2, libxml2 and zlib cross-built for capstone64 on musl-capstone | each library's own tests pass natively and it links for capstone64; the GLib pointer-is-a-word sites patched, each with a reason; the idiom census (item 3) run over the libraries |
| **M0** | the minimal tshark cross-configured and linked as a domain | the image links; its `code_len` and block size are measured |
| **M1–M5** | staged images: M1 `main`, M2 `epan_init` (the heap measured here), M3 capture opened, M4 first frame dissected, M5 all frames | M5's `-V` output byte-identical to stock on all four captures; the flipped control fires; the harness negative control differs |
| **Safety** | the three heap arms from the FFmpeg port, plus wmem's hooks in `sublet` mode | pre-registered fixtures for heap overflow, use after free, stale free, and a stale pointer into a reset packet pool (`block_fast`) and a reset file scope; every predicted fault faults and every control returns, on QEMU, with no fixture on an absent-handle path |

**Auditors** after M-infra and after the safety run, as before this plan.

## Risks

1. **GLib provenance.** capstone64's `uintptr_t`/`gsize` are 64-bit, and pointers are 16
   bytes. GLib says so in ten static assertions, and its integer macros are unmeasured. CHERI's
   GLib ports met the same assertions, because `long` is 64-bit there too; they are the prior
   art to read before patching.
2. **Size, heap and load time:** a 64–128 MiB block; 250k capability initialisers; QEMU TCG
   dissection time.
3. **Upstream assumes the full dissector set.** Three NULL paths are patched and more are
   known. More can appear on the capstone64 path.
4. **wmem's reuse is the temporal story.** A system heap that revokes on `free` sees nothing of
   a rewound `block_fast`.

## Where the work lives

- `capstone/ports/wireshark/app/`:
  - `upstream.json` (the pin);
  - `dissector-whitelist.txt`;
  - `patches/0001-0003`;
  - `src/capstone-stubs.c`.
- The census builds and scripts: `/tmp/capstone/tshark-app/` (not committed).
- This plan.
