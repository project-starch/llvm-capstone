# tshark as a full application in a Capstone domain — M0 census and port plan (2026-09-23)

**Status (2026-09-25):** M-infra, M-deps, M0 and M1–M5 are done on QEMU, and so is the heap half of
Safety. All three heap arms ran every pre-registered fixture three times, all as predicted:
- **level0:** no heap safety;
- **shrink:** spatial safety for g_malloc'd objects;
- **sublet:** spatial and temporal safety for g_malloc'd objects.
- **On every arm:** wmem's allocations stay unprotected. Safety's other half, the wmem hooks, is
  the lead's call.

tshark's output matches stock on all three arms. See "Progress on the full port" below.
- **The tshark domain:** 66.3 MiB, in a 128 MiB CMA block. It reaches all five stages. Its
  `-V -n` output is byte-identical to native stock tshark on the four workload captures, their
  flipped copies and dns-ooo.
- **Evidence:** `ports/wireshark/app/results/2026-09-24-qemu-tshark-staged/`.

The plan and M0 census below are as written before implementation. An adversarial audit ran on
them; its corrections are folded in, and the three claims it refuted are listed under
"Withdrawn" so the trail survives.

**Question:** can tshark (Wireshark 4.6.8) read a capture file and print its full dissection
inside one Capstone domain, and what does it cost? In particular, what does it cost with the
heap **bounded per object and revoked on free**? The FFmpeg port showed that correctness under
enforcement and safety are different results
(`ports/ffmpeg/app/results/2026-09-23-qemu-safety/`), so safety is part of acceptance here
from the start.

**Scope:** QEMU only. Board work is a later, separate question: this ABI depends on QEMU's
fabricated `gp`, and the deployed silicon lets a stale data access retire (ISSUES Q-11).

**Every number below comes from a run on 2026-09-23** unless it is marked "computed" or dated
2026-09-24. The census builds are throwaway and live under `/tmp/capstone/tshark-app/`. What the
census could not settle is marked **UNRESOLVED**. The eight items the audit left open were worked
on 2026-09-24: their results and the go/no-go are in "M0 open items: results" below, and where
they supersede an earlier statement in the census, that statement says so.

## The answer in one paragraph

A minimal tshark works natively. It has 19 whitelisted dissectors, 9 stubs, and 3 empty
tables, and its full `-V` dissection of four shipped captures (10 frames) is byte-identical to
stock tshark's. The comparison can fail: it DIFFERS on six other shipped captures whose
protocols the whitelist leaves out.

Porting it into a domain is a different order of work from FFmpeg (updated 2026-09-24):
- **Image and heap:** tshark and its three Wireshark libraries are 16.2 MiB natively, but only
  1.6 MiB of that is machine code. The rest is read-only data (4.7 MiB), pointer tables (2.8 MiB) and
  dynamic-link tables (6.2 MiB) that a static domain image does not carry.
  - **Size (computed):** about **57–64 MiB** as ported, or **33–42 MiB** with smaller wmem
    arenas. The block is 64 or 128 MiB, depending on that patch and on a `.capstone_domreq`
    declaration; either way it is far over today's 4 MiB ceiling, so it has to come from CMA.
  - **The heap term:** 27.8 MB natively, and 27.3 MB of it is fixed-size wmem arenas (8 MiB and
    2 MiB blocks). Their actual fill is about 1.5 MB.
  - **The initialiser term:** 257k pointer initialisers, which become about 12 MiB of
    initialiser code (computed per table from measured costs).
- **Libraries:** GLib, libgcrypt (with libgpg-error), c-ares, PCRE2 and **libxml2** are all
  required by 4.6.8, plus zlib.
- **Compiler blockers in the linked tshark sources: none left.**
  - `CF_FUNC` is fixed in one line (patch 0003).
  - `__thread` (ISSUES C-47) is compiled away for a single-threaded domain (patch 0004). That
    turns 3 codegen failures into 0.
  - The 6 remaining failures are all an ICU header pulled in by libxml2's configuration. With
    ICU disabled in that configuration they compile (audit).
  - `manuf.c`, the largest table, compiles too: 0 failures, in 91 minutes with a debug clang.
- **Provenance: tshark itself is clean on its linked path.** Of 95 lossy pointer→integer casts
  it has one real round trip, and tshark never reaches it (only sharkd does).
- **GLib is where the work is:**
  - 20 of its 94 core files do not compile for capstone64, 10 of them on its own assertions that
    a pointer is word-sized.
  - Three idioms lose provenance:
    - the once-init macro's `guintptr` cast;
    - `gdataset`'s flag bits kept in a pointer, which tshark's link never pulls in;
    - `gqsort`'s merge sort, which copies 16-byte elements as two `guintptr` halves. The cast
      census cannot see it, and the audit found it by reading.
  - The 9 files that stop on a missing header (futex, wait) were never examined. All 9 are in
    tshark's link.

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
- **Corrected 2026-09-24:**
  - **"`ui/voip_calls.c:145` is not linked into tshark."** It **is** linked
    (`linked-sources.txt`). It is unreached: its only entry, `voip_calls_init_all_taps`, is
    called only from `sharkd_session.c:3896,3953`.
  - **"Pointer-typed `__thread` fails."** ANY real use of `__thread` fails codegen, whatever
    its type and at every `-O` level (`Cannot select: GlobalTLSAddress`). A `__thread` compiles
    only when no TLS address survives to instruction selection: when it is never referenced, or
    when it is `static`, never written, and its reads fold to the initial value at `-O1` and
    above. The `int __thread` sample that seemed to compile was the second case. External,
    written or address-taken variables fail, and so does everything at `-O0` (a matrix checked
    2026-09-24, with the compiler lane).
  - **So a compiling `__thread` is not a stable signal to gate on.** The case that compiles
    breaks as soon as a later patch writes the variable or takes its address, and a build at
    `-O0` breaks it without any patch. The gate compiles at `-O1`. This port does not depend on
    that case: patch 0004 removes every `__thread` from the build.
  - **"The domain block is 64–128 MiB until the heap is measured."** Superseded by the
    measurement (results, item 4).
- **Corrected by the M0-open audit (2026-09-24), after they were reported in session:**
  - **"About 70–86 MiB, a 128 MiB block."** The pointer-width growth factor was applied to
    `size`'s 12.8 MiB "text". That figure is mostly read-only data and 6.2 MiB of dynamic-link
    tables, not code, so relocations were counted twice: once scaled, and again as the
    initialiser code that replaces them. It also assumed the wmem arenas could fill and spill,
    but their fill measures about 1.5 MB. Recomputed from a per-section breakdown in results,
    item 4.
  - **"manuf.c's initialiser needs about 0.8–1.0 MiB of stack; the frame grows linearly per
    initialiser."** The frame tracks DISTINCT materialised targets, not relocations. The
    synthetic table had only distinct strings; real manuf slices cost 2.8–3.5 bytes each
    (results, item 5).
  - **"In the 74 GLib files that compile: 25 lossy sites, including gdataset's."** 10 of the 25
    (and gdataset, ghash and gatomic) are in files that FAIL on static assertions, whose casts
    clang still reports. The 74 that compile hold 15.
  - **"Two provenance-losing GLib idioms."** Three: `gqsort` copies pointer-sized elements
    through `guintptr`. It is a cast of a pointer's type, not of its value, so the cast census
    is blind to it.

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
  - `HOME`/`WIRESHARK_CONFIG_DIR` are pinned to a fresh empty directory per run by
    `host/oracle.sh` (2026-09-24; results, item 7).
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

- **`size`'s "text" is not code.** Berkeley `text` counts every read-only section: code is
  1.64 MiB of the 12.8, and the rest is `.rodata` and dynamic-link tables. The per-section
  breakdown is in results, item 4b.
- **What the native total includes and excludes:**
  - It **includes** 5.8 MiB of `.rela.dyn`, which a static image does not carry. But its 253,969
    relative relocations are pointer initialisers, and each becomes a capability to initialise
    at load. Their cost is measured in results, item 5.
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
- **Estimate:** replaced by results, items 4 and 5. The image is measured at M0 and the heap
  at M2 (after `epan_init`).
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
  single-threaded domain can define it empty: patch 0004 (results, item 2).
- **Codegen pass (audit):** `-S` over all 400 linked sources. 11 fail:
  - 6 on libxml2 headers;
  - 3 on `__thread`;
  - 2 did not finish in 600 s under 48-way load: `epan/manuf.c` with its 4.4 MB table, and
    `pci-ids.c`.

  Superseded by the codegen gate (results, item 1).
- **GLib-macro round trips:** measured with an instrument that sees them (results, item 3).

**GLib 2.80.5** (`glib/`, `gmodule/`, `gthread/`: 96 files, `-fsyntax-only` for capstone64).
The first run was void: 93 files missed `gversionmacros.h`, which meson generates at build
time. After a native build, **20 of 96 do not compile.** (The 96 are compile-database entries:
94 files, two of them compiled twice. The codegen gate counts 94.)
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
- **`__thread` in the dependencies:** checked in results, item 2.

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
- **`musl-capstone/runtime/sublet_heap.c`** (on `dev` since `5e7157e`): the system heap bounded per
  object and revoked on free.
- **The FFmpeg port's shape:** staged images that return, a pristine-built oracle, controls,
  pre-registered safety fixtures, and a per-section verdict.

## M0 open items: results (2026-09-24)

The audit left eight items open before implementation. Each is settled below, or marked
UNRESOLVED with its reason. The census builds are the ones described above: `src-min`,
with patches 0001–0005 applied, and the native builds `native-min` and `native-min-pa`.

### 1. Codegen gate (`host/codegen-gate.py`)

**What it does.** It compiles every linked translation unit for capstone64 with
`-S -o /dev/null`, in codegen and not `-fsyntax-only`.
- **Flags:** each file keeps its native `-I`/`-D`, taken from the compile-database entry that
  produces its object. The capstone64 `config.h` and `glibconfig.h` go first.
- **Exit status:** 1 on any failure or timeout; ERROR (2) if nothing compiled or `CAPSTONE_CLANG`
  is unset (both checked).

**Its controls:**
- **Fires:** a pointer-typed `__thread` sample fails it.
- **Sees the cast classes:** an idiom sample trips both cast classes, 4 sites each.
- **Nothing to compile:** an empty source list is an ERROR, not a pass.
- **Its first run was void and is not counted.** 10 files failed on a harness include path
  (`ws_log_defs.h`). The rerun fixed the path, and the gate now prefers the entry that produces
  the object: one without `-o` had compiled 13 files without `-DNDEBUG`.

**Results.**

| translation units | result |
|---|---|
| **399 linked** (the source list's 401, less `tools/lemon/lemon.c` and `lempar.c`, which are build-time tools) | |
| 392 | compile |
| **6** | fail, all on `libxml2/encoding.h` → `unicode/ucnv.h` (`epan/epan.c` and 5 wiretap readers): the host libxml2 is configured with ICU. With `LIBXML_ICU_ENABLED` removed from a copy of `xmlversion.h`, all 6 compile, 0 failures (audit). M-deps builds libxml2 without ICU |
| `pci-ids.c`, `enterprises.c`, `services.c` | compile, in 1,083 s, 620 s and 234 s (16-way load) |
| **`epan/manuf.c`** | compiles, 0 casts, in 5,474 s (a serial run, 4 h budget); its object's initialiser is measured in item 5 |

**manuf.c's compile time.** Compile time grows faster than linearly with a file's
initialisers. A synthetic table shaped like manuf's took 115 s for 8,000 initialisers and 451 s
for 16,000; the real tables do not follow one law (item 5). manuf.c took 5,474 s for its -S run and 5,540 s
for its -c build (debug clang). That is a build-time cost, not a blocker.

### 2. `__thread` (patch 0004)

**The patch.** `WS_THREAD_LOCAL` is empty when `CAPSTONE_SINGLE_THREAD_DOMAIN` is defined. The
domain is single-threaded: `epan_init` runs registration inline (`tshark.c:1370`), and a native
run makes 0 `clone` calls.

**The compiler fix is merged (2026-09-25).** C-47's fix is local-exec TLS on `tp`'s capability,
with a runtime `tls.c`. It was branch `compiler/c47-tls` (`8abb7757fbf2`; first cited here as
`07829e435e0d`, before the branch was rebased) and was merged into dev as `3979abd8e9a3`.
This port has not yet been rebuilt with it, so the define stays for now. Patch 0004 can go
after one run on the new compiler shows the three files' output unchanged.

**Checked from both sides.** The three `__thread` files (`except.c`, `wtap.c`,
`filesystem.c`) fail the gate without the define, each with `Cannot select:
GlobalTLSAddress`. With the define they compile.

**The dependencies:**

| dependency | `__thread`/`_Thread_local`/`thread_local` |
|---|---|
| libgcrypt 1.11.1 | `src/fips.c:76`, a static `__thread` struct (the file has an `#error` without TLS). It needs the same single-thread patch in M-deps |
| c-ares 1.34.5 | none (a keyword grep of the sources; the same grep finds libgcrypt's) |
| libgpg-error 1.55 | none (same grep) |
| GLib 2.80.5 | defines `G_THREAD_LOCAL`, but no `glib/*.c` uses it |
| PCRE2, zlib, libxml2 | not checked: their versions are not pinned yet (M-deps) |

### 3. Pointer→integer casts, with an instrument that sees them

**The instrument.** The gate counts clang's own `-Wvoid-pointer-to-int-cast`/
`-Wpointer-to-int-cast` (lossy) and `-Wint-to-void-pointer-cast`/`-Wint-to-pointer-cast`
(rebuilt). On capstone64 these fire on every pointer→integer cast, because no integer type the
code casts to is as wide as a 16-byte pointer. Unlike `-Wcapstone-pointer-roundtrip`, they see
GLib's `GPOINTER_TO_*` macros.

**tshark's linked sources: 95 lossy sites** (91 distinct lines) and 196 rebuilt. Each lossy
line was classified by reading it:

| class | lines | verdict |
|---|---:|---|
| a real pointer→integer→pointer round trip that is dereferenced | 2, `ui/voip_calls.c:145,150` (`tap_base_to_id`/`tap_id_to_base`) | loses provenance, but is **unreached in tshark**: its only entry, `voip_calls_init_all_taps`, is called from `sharkd_session.c` alone |
| hashing a real pointer | 2, `epan/stream.c:78,197` | benign |
| alignment test | 1, `epan/in_cksum.c:89` | benign |
| an integer carried in a pointer and read back | 86 | benign, as long as it is only an integer; that is the pattern in every one of them. It covers proto/hf ids in wmem lists, ports in `p_get_proto_data`, uint keys of `wmem_tree`/`wmem_map`/`GHashTable`, and compare functions of integer keys. The wmem lines are `wmem_tree.c`'s uint-keyed lookups (23) and `wmem_miscutl.c:35,41`'s integer compare functions (2), which nothing linked calls |

The rebuilt direction: `GUINT_TO_POINTER` 134, `GINT_TO_POINTER` 50, `GSIZE_TO_POINTER` 2 (the
voip pair), and 10 macro-wrapped `GUINT_TO_POINTER`s (`GET_OPTION_TYPE`, `ENUM_KEY`).

The six libxml2 files, compiled without ICU (audit), add 1 lossy site: `wiretap/ttl.c:199`, an
integer read back from a uint-keyed table, benign. They also add 20 rebuilt sites. None of
these is in the counts above.

**GLib 2.80.5**, 94 files. 74 compile. Of the 20 that fail:
- **11 stop on a non-fatal error:** the 10 pointer-is-a-word static assertions, and
  `gdatetime.c`'s `ALTMON_1`. clang still parses the rest of each file, so their casts ARE
  counted.
- **9 stop on a missing header, so nothing in them was examined:** `linux/futex.h` in 8
  (`gbitlock`, `gcharset`, `gconvert`, `genviron`, `gmessages`, `gthread`, `gthread-posix`,
  `gthreadpool`), and `linux/wait.h` in `gmain.c`. **All 9 are in tshark's static link closure**
  (audit).

The counted sites: 25 lossy and 35 rebuilt. The 74 files that compile hold 15 of the lossy and
20 of the rebuilt; the rest are in the 11 assertion-failing files.

| class | sites | verdict |
|---|---|---|
| once-init: `g_once_init_leave_pointer` is `(gpointer)(guintptr)(result)` (`gthread.h:292,303`) | 3 (`gutils.c:782,1117`, `gtestutils.c:920`) | **loses provenance.** The stored pointer is rebuilt from an integer, and the fix is to drop the cast. More users are among the 9 unexamined files (`gcharset.c`, `gmain.c`, `gthread.c`) |
| `gqsort.c` merge sort | 3 counted (`:268,273,276`, alignment tests) | **loses provenance, and the census cannot see how** (audit, confirmed by reading). For a 16-byte element the alignment test picks copy mode 2 (`:275-277`), which moves each element as two `guintptr` words (`:131` onward). That splits every capability and drops its tag. Reachable: tshark imports `g_ptr_array_sort` and `g_array_sort_with_data`, called from display filters (`epan/dfilter/dfvm.c:841`, `dfilter.c:889`) and `epan/stats_tree.c:1503`. Not called on the four-capture workload (an `LD_PRELOAD` spy with a firing control recorded 0 calls) |
| `gdataset.c` flag bits in a pointer (`G_DATALIST_CLEAN_POINTER`/`SET_POINTER`, `:72,:82`) | 6 (an assertion-failing file) | **loses provenance** (masks and ORs through `guintptr`), but is **not in tshark's link.** A static-link closure from tshark's 319 imported GLib symbols pulls 62 of 95 objects, and neither `gdataset.c.o` nor its only users (`gscanner.c.o`, `glib-private.c.o`) is among them (audit) |
| `gatomic.c:510` `g_atomic_pointer_add` returns the old pointer as `gintptr` | 1 (an assertion-failing file) | an integer result by API; `gatomic.c.o` is not in tshark's link |
| `ghash.c:348,366` small-array compaction | 2 (an assertion-failing file) | **benign in a capstone64 build.** GLib enables it only for pointers ≤ 8 bytes (`ghash.c:196`). With the capstone64 `config.h` (`SIZEOF_VOID_P 16`), preprocessing shows `is_big = TRUE` forced and the path dead; with the native `config.h` it is live. The warning is syntactic. This withdraws the census note that called it "conditional on 32-bit addresses" |
| hashing, integers in pointers (quarks, fds, unichars), alignment tests that choose nothing | the rest | benign |

**What the census cannot see:** a copy that moves pointers through a pointer-sized INTEGER type
(`gqsort`'s case) is a cast of a pointer's type, not of a pointer's value, so no
pointer-to-integer warning fires. The 9 unexamined files, and every such typed copy, need
reading in M-deps.

### 4. Heap (patch 0005, and a byte counter)

**The patch.** 0005 makes `PROTO_PRE_ALLOC_HF_FIELDS_MEM` a build parameter
(`CAPSTONE_HF_PREALLOC`). The minimal build registers 2,202 fields; with 4,096 the oracle still
passes (item 7).

**The counter.** An `LD_PRELOAD` counter tracks live `malloc` bytes and their peak. Its positive
control: an added 25 MiB allocation moved the peak by 26.2 MB.
- Its large-block accounting ignored `calloc` and `realloc`.
- The audit's independent counter covers every allocation entry point and records backtraces.
  It reproduced the composition below to the byte, and found no large `calloc`, `realloc` or
  aligned allocation.

**The native peak heap, over all four captures:**
- **46.9 MB** as shipped;
- **27.8 MB** with `CAPSTONE_HF_PREALLOC=4096`.

**Its composition at the peak: 27.27 MB of the 27.77 MB is in allocations of at least 1 MiB.**
- Over the run the counter sees four 8 MiB blocks (`WMEM_BLOCK_SIZE`,
  `wmem_allocator_block.c:142`) and one 2 MiB `block_fast` block (`wmem_allocator_block_fast.c:42`).
- At the peak, exactly three 8 MiB blocks and the 2 MiB block are live: 27,267,080 =
  8,392,688 + 2 × 8,388,616 + 2,097,160.
- All five come from wmem allocators created in `epan_init`/`guids_init`/`epan_new` and in
  dissection (`epan_dissect_run_with_taps`), through `g_malloc` (backtraces, audit).
- Everything else is about 0.5 MB.

**The arenas' fill: about 1.5 MB.** `WIRESHARK_DEBUG_WMEM_OVERRIDE=simple`
(`wsutil/wmem/wmem_core.c:176-188`) sends every wmem allocation through `malloc`.
- **Peak live heap:** 1.47–1.56 MB over the four captures (audit; reproduced on dhcp and arp:
  1,500,504 and 1,467,304).
- **Output:** byte-identical to the normal run.

So about 95% of the native peak is reserved, empty arena.

**The capstone64 heap (computed):**
- **As ported: about 27 MiB.** The arenas are fixed-size, and even a doubled fill (about 3 MB)
  cannot spill one.
- **About 3–5 MiB** with `WMEM_BLOCK_SIZE` (a compile-time constant) reduced to 1 MiB for the
  domain build.
- It is measured at M2 (after `epan_init`).
- **A conflict for the safety arms:** `sublet_heap.c` (on `dev` since `5e7157e`) defaults to a 4 MiB
  pool (`CAPSTONE_SUBLET_HEAP_LOG 22`, `:66-67`). Three 8 MiB arenas cannot fit in it, so the
  sublet arm needs the smaller arenas or a larger pool.

### 4b. The domain block, recomputed (computed; M0 measures `code_len`)

**The native base.** `size -A` over tshark and its three libraries: 16.2 MiB.

| class | native | capstone64 (computed) |
|---|---:|---:|
| machine code (`.text`, `.plt`, `.init`, `.fini`) | 1.64 MiB | × 1.5–1.7 (dhcp and tcp `.text`, less init code) = 2.5–2.8 MiB |
| pointer initialisers | 256,936 relocations | about 12.2 MiB of init code (item 5) |
| `.rodata` | 4.67 MiB | 4.7–5.1 MiB (strings do not grow) |
| `.data.rel.ro` (pointer tables) | 2.79 MiB | × 1.4–2.0 = 3.9–5.6 MiB |
| `.data`, `.bss` | 0.61 MiB | × 1.35–1.45 = 0.8–0.9 MiB |
| dynamic-link tables (`.rela.*`, `.dynsym`, `.got`, …) | 6.18 MiB | 0: a static domain image has none (the initialiser code replaces them) |
| unwind tables | 0.32 MiB | not counted |
| **image** | | **about 24–28 MiB** |

**Adding the rest (computed):**
- the dependencies: 5.3 MiB natively, so 5–8 MiB, unmeasured;
- the heap: about 27 MiB, or 3–5 MiB with smaller arenas (item 4);
- a declared stack of at least 1 MiB (item 5).

**Total: about 57–64 MiB as ported, or 33–42 MiB with smaller arenas.**

**The block under the current module rule (`capstone.c:152-161`):**
- **With a `.capstone_domreq` declaration:** `code_len` + 8 KiB + the declared data, rounded to
  a power of two. That is 64 MiB, and as ported it sits at the edge of 64 MiB.
- **Without one:** `2 × code_len`, which is 128 MiB either way.

**CMA either way.** Today's ceiling is 4 MiB (`__get_free_pages`, order 10).

### 5. Pointer initialisers

**How they work in a domain.** A static pointer initialiser becomes code: the compiler emits one
`__capstone_cap_init` function per translation unit, and `start-musl.S:91-122` calls each one
before `domain_main`.

**Measured on three real translation units and a synthetic table** (capstone64 objects, `-c`;
cost per NATIVE pointer relocation, `R_X86_64_64` outside debug sections):

| object | native relocations | init code | bytes each | init frame (stack) | bytes each | instructions each |
|---|---:|---:|---:|---:|---:|---:|
| `packet-dhcp.c` | 3,867 | 167,312 | 43.3 | 28,160 | 7.3 | 10.8 |
| `packet-tcp.c` | 1,404 | 64,444 | 45.9 | 12,096 | 8.6 | 11.5 |
| `services.c` | 12,728 | 722,404 | 56.8 | 87,136 | 6.8 | 14.2 |
| synthetic, 8,000 entries × 2 strings | 16,000 | 949,504 | 59.3 | 130,320 | 8.1 | 14.8 |
| **`manuf.c`, whole** | 115,856 | 4,971,832 | 42.9 | 322,080 | 2.8 | |

`-O2` gives the same as `-O1`: dhcp's init code is 167,692 bytes with the same frame.

**The linked tshark has 256,936 such relocations.** The audit got the same total by matching
objects through the compile database's `-o` paths instead of by name. The linked binaries'
dynamic pointer relocations total 256,674, and `libwireshark` alone has 253,969.

**93% of the relocations are in four generated tables:**

| table | relocations | consulted on the workload |
|---|---:|---|
| `manuf.c` | 115,856 | yes: `arp.pcap` prints an OUI organisation |
| `enterprises.c` | 66,364 | yes (the census records it consulted under `-n`) |
| `pci-ids.c` | 43,016 | **no**: `pci_id_str`'s only caller is `packet-ncsi.c`, which is not whitelisted |
| `services.c` | 12,728 | built into a hash table at init (`addr_resolv.c:909`) |

**The cost per relocation depends on how many targets are distinct.**
- **Duplicates are materialised once:** dhcp has 2,489 address materialisations for 3,867
  relocations.
- **Real manuf data (audit):** stride-sampled slices of manuf's own table cost **42.0 and 41.7
  bytes** per relocation (4,000 and 8,000 relocations; manuf has 63,552 distinct strings among
  115,856). The audit's flags rebuild the synthetic object exactly, as a control.
- **enterprises** (99% distinct) should cost close to the synthetic 59 bytes.

**Computed:**
- **Init code, per table:**
  - manuf: 4.9 MB at 42 bytes;
  - enterprises: 3.9 MB at 59;
  - pci-ids: 2.5 MB at about 57;
  - services: 0.72 MB, measured;
  - the other 18,972 relocations: 0.8 MB at 43–46.

  That is **about 12.8 MB (12.2 MiB)**, inside the 11–15 MB range that 42–59 bytes each bounds.
  Dropping `pci-ids.c` removes about 2.5 MB.
  - **The link cannot drop it by itself.** The domain link script KEEPs `.capstone_cap_init`
    (`capstone/my_first_domain/link.ld:82`). Each entry calls its unit's init function, and that
    function stores into the table, so `--gc-sections` keeps every table that has pointer
    initialisers, referenced or not.
  - **So the patch has to take it out of the source list.**
- **Startup:** about 3M instructions, negligible on either platform.
- **Stack: the init frame tracks DISTINCT materialised targets**, which are spilled, not
  relocations. With all-distinct targets the cost is about 7–9 bytes each, linear from 1k to 16k
  (synthetic: 10,176 → 130,320 bytes). Real manuf slices cost 3.5 and 2.8 bytes each, about
  2.2 bytes at the margin (audit).
  - **manuf.c, measured whole:** 4,971,832 bytes of init code (42.9 per relocation, as the
    audit's slices predicted) and a 322,080-byte frame (2.8 per relocation).
  - enterprises.c, nearly all distinct, is probably the largest frame, at about 0.5 MB
    (unmeasured).
  - Declaring **at least 1 MiB** of stack is safe on every reading; the FFmpeg domains declare
    256 KiB.

**For the compiler lane (a cost, not a blocker):** every initialiser is materialised, spilled to
a stack temporary, reloaded and then stored (dhcp: 6,352 `stc`, 3,098 `ldc`). A table-driven
lowering, with entries processed by a loop in `start-musl.S`, would replace about 12 MiB of code
with a data table a few times smaller and remove the frame. It is recorded here and not
changed by this port.

### 6. `NDEBUG`

Every object tshark links is compiled `-DNDEBUG` natively. The 13 compile-database entries
without it produce no object; they belong to other rules. The domain build pins `-DNDEBUG`,
and the gate now compiles each file with its object-producing entry. So the `"file"` handle
path behind `ws_assert` (`packet.c:264-265`) stays disabled, as it is natively.

### 7. Oracle (`host/oracle.sh`)

`HOME` and `WIRESHARK_CONFIG_DIR` are pinned to a fresh empty directory per run, with `TZ=UTC`
and `-V -n`.

**It PASSES only if all of these hold:**
- the 4 workload captures MATCH stock;
- each flipped capture changes stock's output, and the candidate still MATCHES stock on it;
- `ntp.pcap` DIFFERS (the harness's negative control);
- `dns-ooo.pcap` MATCHES (a covered protocol outside the workload).

**Results:**
- **`native-min-pa` (patch 0005 at 4,096): ORACLE PASS.** Each flip changes 1 line.
- **Self-test, stock against stock: ORACLE FAIL**, on the negative control, as it must.

### 8. Absent handles

Handles that whitelisted code looks up but that are never registered in this build:
- `ipx`, `ccsds` (`packet-ieee8023.c:122,124`);
- `http2` (`packet-http.c:5113`);
- `tls-echconfig` (`packet-dns.c:6298`);
- `tpkt` (`prefs.c:5947`);
- `file` (`packet.c:258`, behind `ws_assert`, item 6);
- `sport` (`packet-tcp.c`, guarded).

**No safety fixture may route into one of these paths.** The fixtures are written in Phase 3,
and this list goes into a comment beside them.

## Go / no-go for the full port (2026-09-24)

**Recommendation: GO, conditional on M-infra and the lead's OK.** Nothing found stops the port,
and what it costs is now measured or bounded. An adversarial audit tried to break each pillar
with independent instruments: a second heap counter, a relocation recount, real manuf data, a
GLib link closure, and an ICU-free compile. None of those found a blocker. Its corrections are
folded in above and listed under "Withdrawn".

1. **Compiler:** no blocker in the 398 linked translation units that finished, plus the six
   libxml2 files compiled without ICU.
   - C-47 is avoided by patch 0004.
   - `manuf.c` compiles too (item 1), in 91 minutes.
   - Every compile time here comes from a debug build of clang.
2. **Hard prerequisite: M-infra.** The block cannot come from the buddy allocator (a 4 MiB
   ceiling).
   - **Size (computed):** about 57–64 MiB as ported, or 33–42 MiB with smaller wmem arenas.
     That is a 64 MiB block with a `.capstone_domreq` declaration, and 128 MiB without one
     (item 4b). M0 measures `code_len`.
   - **Governance:** CMA allocation in the kernel module is shared infrastructure, reviewed
     separately. It is the lead's call.
3. **Most of the work is M-deps, and within it GLib:**
   - 20 failing files: 10 pointer-is-a-word assertions, 8 `linux/futex.h`, `linux/wait.h`,
     `ALTMON_1`.
   - Three provenance-losing idioms:
     - once-init, in tshark's link;
     - `gqsort`, in tshark's link and reachable from display filters;
     - `gdataset`, not in tshark's link.
   - The 9 header-failing files are unexamined, and all of them are in tshark's link.
   - Typed copies through `guintptr` need reading, because the census cannot see them.
   - libgcrypt needs the single-thread patch, and libxml2 a build without ICU.
4. **tshark itself is clean on its linked path:** 1 real round trip, unreached; everything else
   benign.
5. **Costs to carry, not stops:**
   - about 12 MiB of initialiser code;
   - a declared stack of at least 1 MiB;
   - hour-scale builds of the four generated tables with the current compiler;
   - the sublet heap arm's 4 MiB pool, which the unpatched wmem arenas do not fit.

## Progress on the full port (2026-09-24)

**M-infra: done on QEMU, in review.**
- **The CMA allocation** already existed: caplifive-buildroot `2b8ad05`, by the external
  collaborator, merged as `7440cfc`, PR #4. The parent still points at `d04bd83`.
- **The port's gate found one defect in it.** A declared domain whose size lands just under a
  power of two gets less `dom_data` than it declared: the monitor's split granule reaches 64 KiB
  at 64 MiB, and the slack is 8 KiB.
- **The fix, `a74a856`** (branch `modcapstone/cma-domain-block`, pushed for review):
  - sizes by the monitor's own arithmetic;
  - refuses a corrupt declaration;
  - leaves blocks of 4 MiB or less unchanged.
- **Evidence:** `ports/wireshark/app/results/2026-09-24-qemu-cma-domain-block/`: a matched pair
  at 64 MiB, 128 MiB domains from CMA, and both refusals.
- **Open, and the lead's call:** bumping the parent's submodule pointer rebuilds every lane's guest
  image.

**M-deps: all seven libraries build and pass their gates.** `ports/wireshark/app/deps/`, one
recipe each, gated as in its README. What that took:
- **zlib, PCRE2, libgpg-error, libxml2:** no provenance-losing site.
- **c-ares:** five "cast off const" round trips, patched.
- **libgcrypt:**
  - three pointers rebuilt from integers, patched;
  - `fips.c`'s `__thread`, now static storage.
- **GLib (libglib only):** cross-configured by its own meson, so `config.h` is answered by this
  libc. That alone removed the census's futex/wait failures. Six patches, all
  `__CAPSTONE__`-guarded except one refactor its native suite covers:
  - `gintptr` holds an address, because no integer type is pointer-wide on capstone64;
  - once-init on a `gsize` uses `gsize` atomics;
  - the pointer bit operations (`and`/`or`/`xor`, `gdataset`, `gbitlock`) move the capability's
    address;
  - four size checks become "large enough";
  - an aligned allocator is built from `malloc`;
  - `gqsort` copies pointer-sized words whole.

**M0: done.** `host/cross-build.sh` builds the minimal tshark with Wireshark's own CMake, 397 of
397 steps. `host/build-domain.sh` links it with the port's own link.
- **Size:** 66.3 MiB. Without the 40 MiB level0 arena that is 26.3 MiB, below this plan's 29–36 MiB
  estimate. The block is 128 MiB.
- **Link gates:** no undefined symbol; no undefined weak symbol, after glib-0007 fixed GLib's two
  LeakSanitizer hooks (ISSUES C-56's open half); the negative control fires.

**M1–M5: done on QEMU.** Evidence: `results/2026-09-24-qemu-tshark-staged/`.
- **Stages and oracle:** every stage returns. M5's `-V -n` stdout is byte-identical to stock on
  dhcp, dns_port, http and arp, on their flipped copies (each flip changes stock's output) and on
  dns-ooo. Its stderr is identical to the native minimal build's. ntp differs, as the negative
  control must.
- **Heap:** the level0 peak is 26.8 MiB.
- **It took three runtime fixes this plan did not foresee,** each port-local:
  - constructors and destructors (`.init_array` never ran in any domain, and musl's exit walked
    `.fini_array` through integers);
  - GCond (musl-capstone's `pthread_cond_t` is too small for its own fields, glib-0008);
  - the unserved-syscall report, which was lost when the program closed fd 1.
  On 2026-09-25 the first and third moved into the shared runtime (ISSUES C-64 and I-11 fixed).
  The GCond one (C-65) awaits the lead's decision on a libc ABI change.
- **Stalls:** one ntp section stalled in the guest before its domain started, the known QEMU stall
  class. Two later ntp runs returned.

**Safety, cheap arms: done on QEMU (2026-09-25).** Evidence:
`results/2026-09-25-qemu-safety/`. Twelve fixtures, pre-registered and pushed before any ran
(`fc2ee56`), on level0 and shrink, N = 3: all 72 counted runs as predicted. One premise in the
predictions file was wrong, though its prediction held: level0 narrows nothing, so its wmem
pointers carry the whole arena, not their block (an audit found it; the README records it).
- **level0:** no heap safety. Every heap pointer carries the whole arena; only the compiler's
  bounds on a global and a stack array fault.
- **shrink:** g_malloc'd objects are spatially exact, and overflow and one-past-the-end fault at
  the printed address. Nothing temporal changes: a stale free still lets a later allocation alias
  a live object.
- **wmem, on both arms:** an allocation carries its whole 2 or 8 MiB block. A stale pointer after
  a scope reset, and an overflow between two wmem allocations, go unnoticed. That is the gap the
  wmem hooks would close.
- **The shrink arm is a working tshark:** M1–M5 and the oracle match stock as on level0.

**Safety, sublet arm: done on QEMU (2026-09-25).** Evidence:
`results/2026-09-25-qemu-safety-sublet/`. Predictions pushed before any sublet boot (`cd06fd2`), and
all 36 counted runs are as predicted.
- **The arm:** the revoking Sublet heap over a 16 MiB pool that the host transfers linear, with
  wmem's blocks cut to 1 MiB (patch 0007).
- **Use after free, use after reuse and a stale free** now fault as temporal, at the printed address.
- **wmem, fixtures 10–12, still return:** a scope reset never calls free.
- **tshark on this heap matches stock:** M1–M5 and the oracle.
- **The node budget:** a full run spends 9,827–13,335 revocation nodes. A boot holds about four such
  runs before capstone-qemu's 65,536-node pool runs out. It ran out once, and QEMU asserted.

**Next, the lead's call:** the wmem hooks (`ports/wireshark/wmem`, `WMEM_PORT_HOOKS`), which would
close the gap all three arms leave. Separately, the port is still built with `b7b31421e9fa`; dev's
compiler is now `3979abd8e9a3` (C-46, C-47), and patch 0004 can go after one run on it.

## Plan

Work lands on `dev` at stable points, as the lead directed on 2026-09-24 ("squash … and then
merge"). The branch `tshark-app` is a local worktree branch and has not been pushed.

| milestone | content | exit criterion |
|---|---|---|
| **M0-open** | the eight items above | each settled or recorded as UNRESOLVED with its reason. Done 2026-09-24 |
| **M-infra** | the domain block from CMA: `capstone.c:163` → `dma_alloc_pages`, `cma=<size>@<base below 4 GiB>` for the QEMU guest, the monitor's rules checked | an existing small domain still runs byte-identically from a CMA block; a 64 MiB and a 128 MiB block allocate and a domain runs in each. Shared infra, reviewed separately, never folded into the port commit |
| **M-deps** | GLib, libgcrypt/libgpg-error (no asm), c-ares, PCRE2, libxml2 and zlib cross-built for capstone64 on musl-capstone | each library's own tests pass natively and it links for capstone64; the GLib pointer-is-a-word sites patched, each with a reason; the once-init `guintptr` cast and `gqsort`'s `guintptr` copy mode fixed (and `gdataset`'s flag bits, if anything pulls it in); the 9 header-failing files read for typed copies; libgcrypt's `fips.c` `__thread` behind the single-thread define; libxml2 built without ICU; the cast census (results, item 3) rerun over the libraries, including GLib's 20 files that do not compile yet |
| **M0** | the minimal tshark cross-configured and linked as a domain, `-DNDEBUG`, `CAPSTONE_HF_PREALLOC=4096`, a declared stack of at least 1 MiB (results, item 5) | the image links; its `code_len`, its initialiser code and its block size are measured against results, items 4 and 5 |
| **M1–M5** | staged images: M1 `main`, M2 `epan_init` (the heap measured here), M3 capture opened, M4 first frame dissected, M5 all frames | M5's `-V` output byte-identical to stock on all four captures; the flipped control fires; the harness negative control differs |
| **Safety** | the three heap arms from the FFmpeg port, plus wmem's hooks in `sublet` mode. The sublet heap's 4 MiB default pool cannot hold three 8 MiB wmem arenas (results, item 4): smaller arenas or a larger pool first | pre-registered fixtures for heap overflow, use after free, stale free, and a stale pointer into a reset packet pool (`block_fast`) and a reset file scope; every predicted fault faults and every control returns, on QEMU, with no fixture on an absent-handle path |

**Auditors** after M-infra and after the safety run, as before this plan.

## Risks

1. **GLib provenance.** capstone64's `uintptr_t`/`gsize` are 64-bit, and pointers are 16
   bytes. GLib says so in ten static assertions. Its casts are counted in 85 of its 94 files
   (three provenance-losing idioms, one found only by reading); the 9 header-failing files are
   unexamined. CHERI's GLib ports met the
   same assertions, because `long` is 64-bit there too; they are the prior art to read before
   patching.
2. **Size, heap and load time:**
   - a 64 MiB block with smaller wmem arenas and a `.capstone_domreq` declaration, or 128 MiB
     without one;
   - about 12 MiB of capability-initialiser code, and a stack frame of up to about 0.5 MiB
     (unmeasured above 16k distinct targets);
   - multi-hour builds of the four generated tables;
   - QEMU TCG dissection time.
3. **Upstream assumes the full dissector set.** Three NULL paths are patched and more are
   known. More can appear on the capstone64 path.
4. **wmem's reuse is the temporal story.** A system heap that revokes on `free` sees nothing of
   a rewound `block_fast`.

## Where the work lives

- `capstone/ports/wireshark/app/`:
  - `upstream.json` (the pin);
  - `dissector-whitelist.txt`;
  - `patches/0001-0005`;
  - `src/capstone-stubs.c`;
  - `host/codegen-gate.py` (the codegen gate and cast census, results item 1);
  - `host/oracle.sh` (the native oracle with its controls, results item 7).
- The census builds and scripts: `/tmp/capstone/tshark-app/` (not committed).
- This plan.
