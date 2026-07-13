/* Minimal SQLite lifecycle harness (agentB-015 CHERI baseline).
 *
 * Upstream SQLite does not execute under CHERI purecap without substantial
 * porting (the amalgamation faults with a misaligned-capability BUS_ADRALN in
 * sqlite3_open at THREADSAFE=0, and deadlocks at THREADSAFE=1 — see RESULTS.md).
 * To isolate the *CHERI* question — does the capability model catch each corpus
 * defect's dangling-pointer access, and when — we compile every corpus shim
 * VERBATIM against this harness, which reproduces the exact allocation, free,
 * callback and handle-invalidation events each CVE depends on. The CHERI verdict
 * depends only on those memory-lifecycle events, not on SQLite's SQL internals.
 *
 * This header mirrors the real SQLite API signatures the shims use, so the shim
 * sources are unmodified. It is NOT a SQL engine.
 */
#ifndef MOCK_SQLITE3_H
#define MOCK_SQLITE3_H

typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
typedef struct sqlite3_context sqlite3_context;
typedef struct sqlite3_value sqlite3_value;
typedef struct sqlite3_backup sqlite3_backup;
typedef long long sqlite3_int64;

#define SQLITE_OK    0
#define SQLITE_ERROR 1
#define SQLITE_BUSY  5
#define SQLITE_ROW   100
#define SQLITE_DONE  101
#define SQLITE_UTF8  1

typedef int (*sqlite3_callback)(void *, int, char **, char **);

int  sqlite3_open(const char *filename, sqlite3 **ppDb);
int  sqlite3_close(sqlite3 *);
int  sqlite3_close_v2(sqlite3 *);
int  sqlite3_exec(sqlite3 *, const char *sql, sqlite3_callback, void *,
                  char **errmsg);
int  sqlite3_prepare_v2(sqlite3 *, const char *sql, int nByte,
                        sqlite3_stmt **ppStmt, const char **pzTail);
int  sqlite3_step(sqlite3_stmt *);
int  sqlite3_reset(sqlite3_stmt *);
int  sqlite3_finalize(sqlite3_stmt *);
const char *sqlite3_column_name(sqlite3_stmt *, int N);
int  sqlite3_column_int(sqlite3_stmt *, int iCol);
const unsigned char *sqlite3_column_text(sqlite3_stmt *, int iCol);

void sqlite3_progress_handler(sqlite3 *, int, int (*)(void *), void *);
int  sqlite3_create_function(sqlite3 *, const char *zFunctionName, int nArg,
                             int eTextRep, void *pApp,
                             void (*xFunc)(sqlite3_context *, int, sqlite3_value **),
                             void (*xStep)(sqlite3_context *, int, sqlite3_value **),
                             void (*xFinal)(sqlite3_context *));
void *sqlite3_user_data(sqlite3_context *);
void  sqlite3_result_int(sqlite3_context *, int);
void  sqlite3_free(void *);

int  sqlite3_set_authorizer(sqlite3 *,
        int (*xAuth)(void *, int, const char *, const char *, const char *,
                     const char *),
        void *pUserData);

sqlite3_backup *sqlite3_backup_init(sqlite3 *pDest, const char *zDestName,
                                    sqlite3 *pSource, const char *zSourceName);
int sqlite3_backup_finish(sqlite3_backup *p);

#endif /* MOCK_SQLITE3_H */
