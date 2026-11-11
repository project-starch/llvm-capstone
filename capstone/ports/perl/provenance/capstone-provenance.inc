/* Private, value-scoped authority for Perl packed pointers.
 * No address lookup, integer-to-pointer conversion, or pointee ownership.
 * Every SV owns its list; snapshots are ordinary mortal SVs, so croak unwinds
 * them using Perl's existing temporary lifetime. Nodes are copied as typed
 * objects, never serialized. Only an explicit value operation transfers them.
 */
#ifdef CAPSTONE_PERL_PROVENANCE
#ifdef EBCDIC
#error "The Capstone provenance adapter currently requires ASCII UTF-8"
#endif
typedef struct cap_prov_entry {
    struct cap_prov_entry *next;
    const char *cap;
    STRLEN off, len;
    U8 bytes[sizeof(char *) * 2];
} cap_prov_entry;

static int cap_prov_free(pTHX_ SV *sv, MAGIC *mg)
{
    cap_prov_entry *p = (cap_prov_entry *)mg->mg_ptr;
    PERL_UNUSED_ARG(sv);
    mg->mg_ptr = NULL;
    while (p) {
        cap_prov_entry *next = p->next;
        Safefree(p);
        p = next;
    }
    return 0;
}

static MGVTBL cap_prov_vtable = { 0, 0, 0, 0, cap_prov_free, 0, 0, 0 };

static MAGIC *cap_prov_magic(SV *sv)
{
    /* Core error paths may retain an arena head marked SVTYPEMASK (0xff).
     * Its former body is no longer an XPVMG, despite the high type value. */
    return sv && !SvIS_FREED(sv) && SvTYPE(sv) >= SVt_PVMG
        ? mg_findext(sv, PERL_MAGIC_ext, &cap_prov_vtable) : NULL;
}

void Perl_cap_prov_clear(pTHX_ SV *sv)
{
    if (cap_prov_magic(sv))
        sv_unmagicext(sv, PERL_MAGIC_ext, &cap_prov_vtable);
}

/* Read only: a mismatch destroys authority instead of healing its bytes. */
static bool cap_prov_valid(SV *sv, const cap_prov_entry *p)
{
    return SvPOKp(sv) && p->off <= SvCUR(sv)
        && p->len <= SvCUR(sv) - p->off
        && memEQ(SvPVX_const(sv) + p->off, p->bytes, p->len);
}

void Perl_cap_prov_trim(pTHX_ SV *sv, STRLEN len)
{
    MAGIC *mg = cap_prov_magic(sv);
    cap_prov_entry *head;
    cap_prov_entry **link;
    if (!mg) return;
    head = (cap_prov_entry *)mg->mg_ptr;
    link = &head;
    while (*link) {
        cap_prov_entry *p = *link;
        if (p->off > len || p->len > len - p->off) {
            *link = p->next;
            Safefree(p);
        } else link = &p->next;
    }
    mg->mg_ptr = (char *)head;
}

void Perl_cap_prov_add(pTHX_ SV *sv, STRLEN off, STRLEN len, const char *cap)
{
    MAGIC *mg;
    cap_prov_entry *p, *head;
    cap_prov_entry **link;
    if (!cap) return;
    if (len > sizeof(p->bytes) || off > SvCUR(sv) || len > SvCUR(sv)-off)
        Perl_croak(aTHX_ "Invalid packed pointer provenance range");
    mg = cap_prov_magic(sv);
    if (!mg) mg = sv_magicext(sv, NULL, PERL_MAGIC_ext,
                              &cap_prov_vtable, NULL, 0);
    /* Any overlap replaces the whole previous field, including equal bytes. */
    head = (cap_prov_entry *)mg->mg_ptr;
    link = &head;
    while (*link) {
        p = *link;
        if (p->off < off + len && off < p->off + p->len) {
            *link = p->next;
            Safefree(p);
        } else link = &p->next;
    }
    /* Allocation can croak. Cleanup must see the already-pruned live list. */
    mg->mg_ptr = (char *)head;
    Newx(p, 1, cap_prov_entry);
    p->cap = cap;
    p->off = off;
    p->len = len;
    Copy(SvPVX_const(sv)+off, p->bytes, len, U8);
    p->next = head;
    mg->mg_ptr = (char *)p;
}

static bool cap_prov_benign(const MGVTBL *v)
{
    return !v || v == &cap_prov_vtable || v == &PL_vtbl_utf8
        || v == &PL_vtbl_mglob || v == &PL_vtbl_taint
#ifdef USE_LOCALE_COLLATE
        || v == &PL_vtbl_collxfrm
#endif
        ;
}

void Perl_cap_prov_prepare_get(pTHX_ SV *sv)
{
    MAGIC *other;
    if (!cap_prov_magic(sv)) return;
    for (other = SvMAGIC(sv); other; other = other->mg_moremagic) {
        const MGVTBL *v = other->mg_virtual;
        if (!(other->mg_flags & MGf_GSKIP)
            && !cap_prov_benign(v) && v->svt_get) {
            Perl_cap_prov_clear(aTHX_ sv);
            return;
        }
    }
}

static SV *cap_prov_snapshot(pTHX_ SV *sv, bool observed)
{
    MAGIC *mg = cap_prov_magic(sv);
    cap_prov_entry *p;
    SV *snap;
    if (!mg) return NULL;
    /* A callback may replace the value independently of the observed bytes.
     * Free-only provenance magic is RMG, not active get/set magic. */
    if (SvAMAGIC(sv)) {
        Perl_cap_prov_clear(aTHX_ sv);
        return NULL;
    }
    if (!observed) {
        MAGIC *other;
        for (other = SvMAGIC(sv); other; other = other->mg_moremagic) {
            const MGVTBL *v = other->mg_virtual;
            if (cap_prov_benign(v)) continue;
            if (v->svt_get || v->svt_set) {
                Perl_cap_prov_clear(aTHX_ sv);
                return NULL;
            }
        }
    }
    for (p = (cap_prov_entry *)mg->mg_ptr; p; p = p->next)
        if (!cap_prov_valid(sv, p)) {
            Perl_cap_prov_clear(aTHX_ sv);
            return NULL;
        }
    if (!mg->mg_ptr) return NULL;
    snap = sv_2mortal(newSVpvn(SvPVX_const(sv), SvCUR(sv)));
    if (SvUTF8(sv)) SvUTF8_on(snap);
    for (p = (cap_prov_entry *)mg->mg_ptr; p; p = p->next)
        Perl_cap_prov_add(aTHX_ snap, p->off, p->len, p->cap);
    return snap;
}

SV *Perl_cap_prov_snapshot(pTHX_ SV *sv)
{
    return cap_prov_snapshot(aTHX_ sv, FALSE);
}

/* Only for a value already evaluated by the caller, or a fresh value being
 * committed by a core setter. prepare_get revokes old metadata before any
 * callback; this never evaluates magic or obtains authority from equality. */
SV *Perl_cap_prov_observed(pTHX_ SV *sv)
{
    return cap_prov_snapshot(aTHX_ sv, TRUE);
}

/* Map byte offsets through exactly the Latin-1/UTF-8 transformation used by
 * Perl. Return false for partial characters or non-byte Unicode values. */
static bool cap_prov_offset(const U8 *s, STRLEN n, int encoding, STRLEN *out)
{
    STRLEN i = 0, count = 0;
    while (i < n) {
        if (encoding > 0) {
            count += NATIVE_BYTE_IS_INVARIANT(s[i]) ? 1 : 2;
            i++;
        } else if (encoding < 0 && !UTF8_IS_INVARIANT(s[i])) {
            if (i+1 >= n || (s[i] != 0xc2 && s[i] != 0xc3)
                || (s[i+1] & 0xc0) != 0x80) return FALSE;
            i += 2;
            count++;
        } else { i++; count++; }
    }
    *out = count;
    return TRUE;
}

void Perl_cap_prov_copy(pTHX_ SV *dst, SV *snap, STRLEN off,
                        STRLEN len, STRLEN to, int encoding)
{
    MAGIC *mg = cap_prov_magic(snap);
    cap_prov_entry *p;
    assert(dst != snap);
    if (!mg || off > SvCUR(snap) || len > SvCUR(snap)-off) return;
    for (p = (cap_prov_entry *)mg->mg_ptr; p; p = p->next) {
        STRLEN start, end;
        U8 expected[sizeof(p->bytes)];
        STRLEN i, n = 0;
        const U8 *raw = (const U8 *)SvPVX_const(snap) + off;
        if (p->off < off || p->off-off > len || p->len > len-(p->off-off))
            continue;
        if (!cap_prov_offset(raw, p->off-off, encoding, &start)
            || !cap_prov_offset(raw, p->off-off+p->len, encoding, &end)) continue;
        if (start > SvCUR(dst) || to > SvCUR(dst)-start
            || end-start > SvCUR(dst)-(to+start)) continue;
        for (i = 0; i < p->len; i++) {
            U8 ch = p->bytes[i];
            if (encoding > 0 && ch >= 0x80) {
                if (n+2 > sizeof(expected)) break;
                expected[n++] = 0xc0 | (ch >> 6);
                expected[n++] = 0x80 | (ch & 0x3f);
            } else if (encoding < 0 && ch >= 0x80) {
                if (++i >= p->len) break;
                expected[n++] = ((ch & 3) << 6) | (p->bytes[i] & 0x3f);
            } else {
                if (n == sizeof(expected)) break;
                expected[n++] = ch;
            }
        }
        if (i != p->len || n != end-start
            || !memEQ(SvPVX_const(dst)+to+start, expected, n)) continue;
        Perl_cap_prov_add(aTHX_ dst, to+start, n, p->cap);
    }
}

const char *Perl_cap_prov_pointer(pTHX_ SV *sv, STRLEN off, STRLEN len,
                                const char *candidate)
{
    MAGIC *mg = cap_prov_magic(sv);
    cap_prov_entry *p;
    const char *zero = NULL;
    if (memEQ(&candidate, &zero, sizeof(candidate))) return NULL;
    if (mg) for (p = (cap_prov_entry *)mg->mg_ptr; p; p = p->next)
        if (!cap_prov_valid(sv, p)) {
            Perl_cap_prov_clear(aTHX_ sv);
            mg = NULL;
            break;
        }
    if (mg) for (p = (cap_prov_entry *)mg->mg_ptr; p; p = p->next)
        if (p->off == off && p->len == len && cap_prov_valid(sv, p)
            && memEQ(&candidate, &p->cap, sizeof(candidate))) return p->cap;
#ifdef __CAPSTONE_PURECAP__
    /* An actual hardware tag is already authority; raw bytes are not. */
    if (__builtin_capstone_cap_get_tag((void *)candidate)) return candidate;
#endif
    Perl_croak(aTHX_ "Cannot unpack pointer without matching provenance");
    return NULL;
}

SV *Perl_cap_prov_newsv(pTHX_ SV *source, const char *pv, STRLEN len, U32 flags)
{
    SV *snapshot = Perl_cap_prov_observed(aTHX_ source);
    SV *result = newSVpvn_flags(pv, len, flags);
    if (snapshot) Perl_cap_prov_copy(aTHX_ result, snapshot, 0, len, 0, 0);
    return result;
}

void Perl_cap_prov_catpvn(pTHX_ SV *dst, SV *source, const char *pv, STRLEN len, I32 flags)
{
    SV *snapshot = Perl_cap_prov_observed(aTHX_ source);
    STRLEN written = len;
    int encoding = 0;
    sv_catpvn_flags(dst, pv, len, flags);
    if (!snapshot) return;
    if ((flags & SV_CATBYTES) && SvUTF8(dst)) {
        encoding = 1;
        if (!cap_prov_offset((const U8 *)SvPVX_const(snapshot), len, 1, &written)) return;
    }
    if (written <= SvCUR(dst)) Perl_cap_prov_copy(aTHX_ dst, snapshot, 0, len,
                                                 SvCUR(dst)-written, encoding);
}
#endif
