/* Original external diagnostic adapter for public printf and BLAS interfaces.
 * Changes console precision and records call arguments/results, not arithmetic.
 * No optimizer/BLAS implementation source is read or included. */
#include <ctype.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t positional(const char *text, size_t at) {
    size_t end = at;
    while (isdigit((unsigned char)text[end])) ++end;
    return text[end] == '$' ? end + 1 : at;
}

static char *precise_format(const char *format) {
    size_t length = strlen(format);
    if (length > (SIZE_MAX - 1) / 3) return NULL;
    char *output = malloc(3 * length + 1);
    if (!output) return NULL;
    size_t used = 0;
    for (size_t at = 0; at < length;) {
        if (format[at] != '%') { output[used++] = format[at++]; continue; }
        size_t start = at++;
        if (format[at] == '%') {
            output[used++] = '%'; output[used++] = '%'; ++at; continue;
        }
        at = positional(format, at);
        while (format[at] && strchr("-+ #0'", format[at])) ++at;
        if (format[at] == '*') at = positional(format, at + 1);
        else while (isdigit((unsigned char)format[at])) ++at;
        size_t precision_start = at, precision_end = at;
        unsigned precision = 0;
        int specified = 0, dynamic = 0;
        if (format[at] == '.') {
            specified = 1; ++at;
            if (format[at] == '*') { dynamic = 1; at = positional(format, at + 1); }
            else while (isdigit((unsigned char)format[at])) {
                if (precision < 17) precision = precision * 10 + (unsigned)(format[at] - '0');
                ++at;
            }
            precision_end = at;
        }
        int long_double = format[at] == 'L';
        if (format[at] && strchr("hlLjztq", format[at])) {
            char modifier = format[at++];
            if ((modifier == 'h' || modifier == 'l') && format[at] == modifier) ++at;
        }
        if (at == length) {
            memcpy(output + used, format + start, at - start); used += at - start; break;
        }
        int floating = strchr("fFeEgG", format[at]) != NULL;
        if (floating && !dynamic && !long_double && (!specified || precision < 17)) {
            memcpy(output + used, format + start, precision_start - start);
            used += precision_start - start;
            memcpy(output + used, ".17", 3); used += 3;
            memcpy(output + used, format + precision_end, at - precision_end + 1);
            used += at - precision_end + 1;
        } else {
            memcpy(output + used, format + start, at - start + 1); used += at - start + 1;
        }
        ++at;
    }
    output[used] = '\0';
    return output;
}

#ifdef RCOMPAT_FORMAT_TEST
int main(void) {
    const char *inputs[] = {"%g", "%8.5e", "%.40g", "%Lg", "%%g", "%*.*g", "%zu %s", "%9.2lf", "%2$g", "%", "%m error", "%a", "%08g"};
    const char *expected[] = {"%.17g", "%8.17e", "%.40g", "%Lg", "%%g", "%*.*g", "%zu %s", "%9.17lf", "%2$.17g", "%", "%m error", "%a", "%08.17g"};
    for (unsigned i = 0; i < sizeof(inputs) / sizeof(*inputs); ++i) {
        char *actual = precise_format(inputs[i]);
        if (!actual || strcmp(actual, expected[i])) return 1;
        free(actual);
    }
    puts("13 format/argument-preservation checks passed");
    return 0;
}
#else
#undef vsnprintf
extern int __vsnprintf_chk(char *, size_t, int, size_t, const char *, va_list);
extern double ddot_(const int *, const double *, const int *, const double *, const int *);
extern void daxpy_(const int *, const double *, const double *, const int *, double *, const int *);

/* Calls from the interposing image retain their original dyld bindings. */
static int precise_vsnprintf(char *target, size_t size, const char *format, va_list args) {
    char *precise = precise_format(format);
    int result = vsnprintf(target, size, precise ? precise : format, args);
    free(precise); return result;
}
static int precise_checked(char *target, size_t size, int flags, size_t capacity, const char *format, va_list args) {
    char *precise = precise_format(format);
    int result = __vsnprintf_chk(target, size, flags, capacity, precise ? precise : format, args);
    free(precise); return result;
}
static int precise_vfprintf(FILE *target, const char *format, va_list args) {
    char *precise = precise_format(format);
    int result = vfprintf(target, precise ? precise : format, args);
    free(precise); return result;
}

static FILE *trace;
static unsigned long sequence;
static FILE *public_trace(void) {
    const char *path = getenv("RCOMPAT_BLAS_TRACE");
    if (path && *path && !trace) {
        trace = fopen(path, "w");
        if (!trace) { perror("BLAS trace"); exit(2); }
    }
    return trace;
}
static void vector(FILE *file, const double *values, int length, int stride) {
    fputc('[', file);
    int start = stride < 0 ? (1 - length) * stride : 0;
    for (int i = 0; i < length; ++i) {
        if (i) fputc(',', file);
        fprintf(file, "\"%a\"", values[start + i * stride]);
    }
    fputc(']', file);
}
static double observed_ddot(const int *n, const double *x, const int *ix, const double *y, const int *iy) {
    double result = ddot_(n, x, ix, y, iy);
    FILE *file = public_trace();
    if (file) {
        fprintf(file, "{\"seq\":%lu,\"fn\":\"dot\",\"n\":%d,\"x\":", sequence++, *n);
        vector(file, x, *n, *ix); fputs(",\"y\":", file); vector(file, y, *n, *iy);
        fprintf(file, ",\"result\":\"%a\"}\n", result); fflush(file);
    }
    return result;
}
static void observed_daxpy(const int *n, const double *alpha, const double *x, const int *ix, double *y, const int *iy) {
    FILE *file = public_trace();
    if (file) {
        fprintf(file, "{\"seq\":%lu,\"fn\":\"axpy\",\"n\":%d,\"alpha\":\"%a\",\"x\":", sequence++, *n, *alpha);
        vector(file, x, *n, *ix); fputs(",\"before\":", file); vector(file, y, *n, *iy);
    }
    daxpy_(n, alpha, x, ix, y, iy);
    if (file) { fputs(",\"after\":", file); vector(file, y, *n, *iy); fputs("}\n", file); fflush(file); }
}

/* Public macOS dyld interposing ABI; function-address pairs, no copied code. */
__attribute__((used, section("__DATA,__interpose,interposing")))
static const struct { const void *replacement; const void *original; } interfaces[] = {
    {(const void *)precise_vsnprintf, (const void *)vsnprintf},
    {(const void *)precise_checked, (const void *)__vsnprintf_chk},
    {(const void *)precise_vfprintf, (const void *)vfprintf},
    {(const void *)observed_ddot, (const void *)ddot_},
    {(const void *)observed_daxpy, (const void *)daxpy_}
};
#endif
