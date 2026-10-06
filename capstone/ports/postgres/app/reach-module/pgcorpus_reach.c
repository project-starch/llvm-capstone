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
 */
#include "postgres.h"

#include "catalog/pg_collation_d.h"
#include "fmgr.h"
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
