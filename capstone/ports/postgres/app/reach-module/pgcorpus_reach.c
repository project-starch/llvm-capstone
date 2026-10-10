/*
 * C-level callers for two PostgreSQL 17.5 defects that no SQL statement can
 * reach.
 *
 * Cases 05 (849da8210539, ascii) and 06 (e91dcfccaa, regexp) both need a text
 * datum that is invalid in the database encoding while the database encoding
 * is multibyte. Those two conditions are mutually exclusive from SQL, which
 * was measured against a native 17.5 cluster on 2026-10-06: a UTF8 database
 * rejects invalid bytes on every route tried, convert_from(bytea,'LATIN1')
 * succeeds but yields valid UTF-8, and a SQL_ASCII database has
 * pg_encoding_max_length 1, which skips both defective branches. Upstream
 * guards one of the two with Assert rather than a runtime check, which is
 * upstream saying the same thing.
 *
 * They are not mutually exclusive below SQL. palloc does not validate, so a
 * caller inside the backend can build the datum and hand it to the defective
 * code -- which is what upstream's own note about already-corrupt on-disk
 * data describes. What these functions measure is therefore the memory safety
 * of the defective code when it is reached, not SQL-level exploitability.
 *
 * Both datums come from palloc, so both defects stay in the aset layer and
 * both cases run on all three arms, unlike the non-nested c-repros.
 *
 * corpus_control, at the end, is not a defect: it is the arm's control set.
 */
#include "postgres.h"

#include "catalog/pg_collation_d.h"
#include "fmgr.h"
/* pgcrypto's own header, for the real PGP_Context: this case turns on where
 * sess_key sits inside that struct and how soon a copy into it leaves it, so
 * a look-alike declared here would measure a layout upstream does not have.
 * Only the header is used -- no pgcrypto object is linked, which is the point,
 * because pgcrypto itself cannot be built for a Capstone domain. */
#include "pgp.h"
#include "mb/pg_wchar.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "varatt.h"

PG_MODULE_MAGIC;

/*
 * Case 05. In a UTF8 database ascii() branches on the lead byte and then reads
 * that many continuation bytes, without ever checking them against the length
 * of the string. A lead byte of 0xF0 claims three. With a payload of one byte
 * the read runs three bytes past the end of the datum's data.
 *
 * payload is the datum's data length in bytes, so payload = 1 is the smallest
 * over-read and larger values move the lead byte's continuation bytes back
 * inside the allocation; a caller sweeps it to find where an arm starts to
 * report.
 */
PG_FUNCTION_INFO_V1(corpus_ascii_truncated_lead);

Datum
corpus_ascii_truncated_lead(PG_FUNCTION_ARGS)
{
	int32		payload = PG_GETARG_INT32(0);
	text	   *string;

	if (payload < 1 || payload > 1024)
		elog(ERROR, "payload must be between 1 and 1024, not %d", payload);
	if (GetDatabaseEncoding() != PG_UTF8)
		elog(ERROR, "this case needs a UTF8 database, not %s",
			 GetDatabaseEncodingName());

	string = (text *) palloc(VARHDRSZ + payload);
	SET_VARSIZE(string, VARHDRSZ + payload);
	memset(VARDATA(string), 0, payload);
	/* selects the branch that reads three continuation bytes */
	*((unsigned char *) VARDATA(string)) = 0xF0;

	PG_RETURN_DATUM(OidFunctionCall1(F_ASCII, PointerGetDatum(string)));
}

/*
 * Case 06. regexp match and split size the buffer they convert wide characters
 * back into by the subject's own byte length, on the stated assumption that
 * re-encoding cannot produce more bytes than came in. That holds for valid
 * input only. pg_mb2wchar_with_len turns each byte that is invalid in the
 * encoding into one pg_wchar, and pg_wchar2mb_with_len spends two bytes
 * putting a value in 0x80..0xFF back, so a subject of n invalid bytes converts
 * back to 2n and overruns the buffer of n + 1 that regexp.c:1592 settled on.
 *
 * The subject is payload bytes of 0x80, each invalid as a lead byte in UTF8.
 */
PG_FUNCTION_INFO_V1(corpus_regexp_invalid_subject);

Datum
corpus_regexp_invalid_subject(PG_FUNCTION_ARGS)
{
	int32		payload = PG_GETARG_INT32(0);
	text	   *subject;
	text	   *pattern;

	if (payload < 1 || payload > 65536)
		elog(ERROR, "payload must be between 1 and 65536, not %d", payload);
	if (GetDatabaseEncoding() != PG_UTF8)
		elog(ERROR, "this case needs a UTF8 database, not %s",
			 GetDatabaseEncodingName());

	subject = (text *) palloc(VARHDRSZ + payload);
	SET_VARSIZE(subject, VARHDRSZ + payload);
	memset(VARDATA(subject), 0x80, payload);
	pattern = cstring_to_text("(.+)");

	/* the regexp functions are collation-sensitive and refuse InvalidOid */
	PG_RETURN_DATUM(OidFunctionCall2Coll(F_REGEXP_MATCH_TEXT_TEXT,
										 C_COLLATION_OID,
										 PointerGetDatum(subject),
										 PointerGetDatum(pattern)));
}


/*
 * Case 09, 7a7d9693c7 (CVE-2026-2005). pgp_pub_decrypt_bytea() takes the
 * session-key length out of the message with no upper bound and copies that
 * many bytes into a 32-byte inline array.
 *
 * The three statements below are pgp-pubdec.c:226-228 verbatim. What is not
 * upstream is how the message gets here: upstream's own regression data
 * arrives through pgp_pub_decrypt_bytea and an RSA or ElGamal decryption,
 * which needs pgcrypto, which needs OpenSSL, which is not cross-compiled for
 * capstone64 -- so on two of the three arms the extension cannot be created
 * at all and the case had no verdict there. Supplying msglen directly keeps
 * the defective arithmetic and the real struct while dropping the cipher the
 * platform lacks, and nothing is lost by it: the overflow at :228 happens
 * before any cipher is instantiated, with ctx->cipher_algo on the line above
 * read but not yet used.
 *
 * msglen is the attacker-chosen length; sess_key_len is msglen - 3. The copy
 * runs off sess_key into sess_key_len, which is the next member, and then off
 * the end of the struct -- on the host that is at 36 bytes copied, with
 * sess_key at offset 148 of 184. The function reports the figures for the
 * platform it is running on rather than leaving a reader to assume the host's.
 */
PG_FUNCTION_INFO_V1(corpus_pgp_sesskey_overflow);

Datum
corpus_pgp_sesskey_overflow(PG_FUNCTION_ARGS)
{
	int32		msglen = PG_GETARG_INT32(0);
	PGP_Context *ctx;
	uint8	   *msg;

	if (msglen < 4 || msglen > 65536)
		elog(ERROR, "msglen must be between 4 and 65536, not %d", msglen);

	ctx = palloc0(sizeof *ctx);
	msg = palloc0(msglen);
	msg[0] = 7;					/* the cipher-algo byte, read at :226 */

	elog(NOTICE,
		 "PGP_Context is %zu bytes, sess_key at %zu is %d wide, "
		 "sess_key_len at %zu; copying %d bytes",
		 sizeof *ctx, offsetof(struct PGP_Context, sess_key), PGP_MAX_KEY,
		 offsetof(struct PGP_Context, sess_key_len), msglen - 3);

	/* pgp-pubdec.c:226-228, verbatim */
	ctx->cipher_algo = *msg;
	ctx->sess_key_len = msglen - 3;
	memcpy(ctx->sess_key, msg + 1, ctx->sess_key_len);

	PG_RETURN_INT32((int32) ctx->sess_key_len);
}

/*
 * The arm's controls, run in the server before any case and on exactly the
 * image being measured (bug-corpora/postgres/sql-repros/shared/run-arm.py).
 * Not defects. Each says what the arm IS, and tools/arms.json records what
 * each must do on each configuration: a write one past a malloc'd object
 * faults wherever malloc bounds objects; a read after free faults only where
 * free revokes; the palloc pair does the same through a memory context, which
 * only the Sublet context pools bound and revoke per chunk. The NOTICE is the
 * mark: it is printed last before the access, so a run that faults without it
 * faulted somewhere else.
 */
static volatile unsigned char corpus_control_sink;

static void __attribute__((noinline))
corpus_control_write(volatile unsigned char *p)
{
	*p = 1;
}

static unsigned char __attribute__((noinline))
corpus_control_read(const volatile unsigned char *p)
{
	return *p;
}

PG_FUNCTION_INFO_V1(corpus_control);

Datum
corpus_control(PG_FUNCTION_ARGS)
{
	char	   *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
	unsigned char *p;

	if (strcmp(name, "bounds-malloc") == 0)
	{
		if ((p = malloc(16)) == NULL)
			elog(ERROR, "malloc failed");
		elog(NOTICE, "CONTROL %s mark", name);
		corpus_control_write(p + 16);
		free(p);
	}
	else if (strcmp(name, "uaf-malloc") == 0)
	{
		if ((p = malloc(32)) == NULL)
			elog(ERROR, "malloc failed");
		p[0] = 7;
		free(p);
		elog(NOTICE, "CONTROL %s mark", name);
		corpus_control_sink = corpus_control_read(p);
	}
	else if (strcmp(name, "bounds-palloc") == 0)
	{
		p = palloc(16);
		elog(NOTICE, "CONTROL %s mark", name);
		corpus_control_write(p + 16);
		pfree(p);
	}
	else if (strcmp(name, "uaf-palloc") == 0)
	{
		p = palloc(32);
		p[0] = 7;
		pfree(p);
		elog(NOTICE, "CONTROL %s mark", name);
		corpus_control_sink = corpus_control_read(p);
	}
	else
		elog(ERROR, "unknown control \"%s\"", name);
	PG_RETURN_TEXT_P(cstring_to_text("CONTROL RETURNED"));
}
