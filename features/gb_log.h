/* SPDX-License-Identifier: GPL-2.0-only
 * gb_log.h - userspace structured-logging helpers.
 *
 * Audit references: AUDIT_2026-05-13.md F-L6-23 (structured JSON logging),
 * F-L6-24 (log-level discipline), F-L6-25 (rate-limited warnings).
 *
 * Includers:
 *   - userspace .c files only (shim, netd, netc)
 *   - kernel module should keep pr_info/pr_warn/pr_err
 *
 * Behaviour:
 *   GREENBOOST_LOG_JSON=1  → one JSON object per log line on stderr
 *   (default)              → human prefix "[level] [greenboost-<component>] msg"
 *
 * Macros:
 *   gb_log_info(component, fmt, ...)
 *   gb_log_warn(component, fmt, ...)
 *   gb_log_error(component, fmt, ...)
 *   gb_log_ratelimit_warn(component, every_secs, fmt, ...)
 *
 * `component` is a short literal - "shim" / "netd" / "netc" / etc.
 *
 * Designed as drop-in for fprintf(stderr, ...) call sites; preserves the
 * `%s` format-string discipline checked by -Wformat-security.  Migration is
 * intentionally manual so each existing call gets re-classified by level.
 */
#ifndef GREENBOOST_GB_LOG_H
#define GREENBOOST_GB_LOG_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- internal helpers ---------------------------------------------------- */

static inline int _gb_log_json_enabled(void)
{
    static int cached = -1;
    int c = __atomic_load_n(&cached, __ATOMIC_RELAXED);
    if (__builtin_expect(c < 0, 0)) {
        const char *e = getenv("GREENBOOST_LOG_JSON");
        c = (e && e[0] == '1') ? 1 : 0;
        __atomic_store_n(&cached, c, __ATOMIC_RELAXED);
    }
    return c;
}

static inline void _gb_log_emit(const char *level,
                                const char *component,
                                const char *fmt,
                                va_list ap)
{
    char buf[1024];
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    if (n < 0) { buf[0] = '\0'; n = 0; }
    if ((size_t)n >= sizeof(buf)) {
        /* Truncated: walk back to a safe ASCII boundary then append marker so
         * JSON consumers never see a half-escaped UTF-8 sequence. */
        n = (int)sizeof(buf) - 4;
        while (n > 0 && (unsigned char)buf[n] >= 0x80) n--;
        buf[n++] = '.'; buf[n++] = '.'; buf[n++] = '.'; buf[n] = '\0';
    }

    time_t t = time(NULL);
    struct tm tm;
#if defined(__GLIBC__) || defined(__APPLE__) || defined(__FreeBSD__)
    localtime_r(&t, &tm);
#else
    tm = *localtime(&t);
#endif
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &tm);

    if (_gb_log_json_enabled()) {
        /* Minimal JSON encoder for the message body (handles quotes + backslashes
         * and the most common control chars).  Not RFC-8259 perfect but enough
         * for log aggregation (Loki, ELK, Datadog) to consume. */
        char esc[1024 * 2];
        size_t ei = 0;
        for (int i = 0; i < n && ei + 2 < sizeof(esc); i++) {
            unsigned char c = (unsigned char)buf[i];
            switch (c) {
            case '"':  esc[ei++] = '\\'; esc[ei++] = '"';  break;
            case '\\': esc[ei++] = '\\'; esc[ei++] = '\\'; break;
            case '\n': esc[ei++] = '\\'; esc[ei++] = 'n';  break;
            case '\r': esc[ei++] = '\\'; esc[ei++] = 'r';  break;
            case '\t': esc[ei++] = '\\'; esc[ei++] = 't';  break;
            default:
                if (c < 0x20) {
                    if (ei + 6 >= sizeof(esc)) break;
                    ei += (size_t)snprintf(&esc[ei], sizeof(esc) - ei,
                                           "\\u%04x", c);
                } else {
                    esc[ei++] = (char)c;
                }
            }
        }
        esc[ei] = '\0';
        fprintf(stderr,
                "{\"ts\":\"%s\",\"level\":\"%s\",\"component\":\"greenboost-%s\",\"msg\":\"%s\"}\n",
                ts, level, component, esc);
    } else {
        fprintf(stderr, "[%s] [%s] [greenboost-%s] %s\n",
                ts, level, component, buf);
    }
}

static inline void _gb_log_impl(const char *level,
                                const char *component,
                                const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    _gb_log_emit(level, component, fmt, ap);
    va_end(ap);
}

/* ---- public macros ------------------------------------------------------- */

#define gb_log_info(component, ...)   _gb_log_impl("INFO",  component, __VA_ARGS__)
#define gb_log_warn(component, ...)   _gb_log_impl("WARN",  component, __VA_ARGS__)
#define gb_log_error(component, ...)  _gb_log_impl("ERROR", component, __VA_ARGS__)

/* Rate-limited warning.  Suppresses repeats of the same call site that fire
 * inside `every_secs` of the previous emission; the suppressed-count is
 * folded into the next emission's payload so SREs can see it spiked. */
#define gb_log_ratelimit_warn(component, every_secs, ...)                          \
    do {                                                                           \
        static time_t        _gb_last       = 0;                                   \
        static unsigned long _gb_suppressed = 0;                                   \
        time_t _gb_now = time(NULL);                                               \
        if (_gb_now - __atomic_load_n(&_gb_last, __ATOMIC_RELAXED) >= (every_secs)) { \
            unsigned long _gb_s =                                                  \
                __atomic_exchange_n(&_gb_suppressed, 0UL, __ATOMIC_RELAXED);       \
            if (_gb_s > 0) {                                                       \
                _gb_log_impl("WARN", (component),                                  \
                             "[suppressed %lu earlier]", _gb_s);                   \
            }                                                                      \
            _gb_log_impl("WARN", (component), __VA_ARGS__);                        \
            __atomic_store_n(&_gb_last, _gb_now, __ATOMIC_RELAXED);                \
        } else {                                                                   \
            __atomic_fetch_add(&_gb_suppressed, 1UL, __ATOMIC_RELAXED);            \
        }                                                                          \
    } while (0)

#endif /* GREENBOOST_GB_LOG_H */
