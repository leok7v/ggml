#pragma once
#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <io.h>
#include <malloc.h>
#include <math.h>
#include <stdarg.h>
// #include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>

// tiny Windows runtime for missing stuff

#ifdef __cplusplus
extern "C" {
#endif

#ifndef countof
    #define countof(a) (sizeof(a) / sizeof((a)[0])) // MS is _countof()
#endif

#define null ((void*)0) // like nullptr but for C99

uint32_t random32(uint32_t* state); // state aka seed
double   seconds();     // seconds since boot (3/10MHz resolution)
int64_t  nanoseconds(); // nanoseconds since boot (3/10MHz resolution)
void     printline(const char* file, int line, const char* func,
                   const char* format, ...);
int      memmap_resource(const char* label, void* *data, int64_t *bytes);
void*    load_dl(const char* pathname); // dlopen | LoadLibrary
void*    find_symbol(void* dl, const char* symbol); // dlsym | GetProcAddress
void     sleep(double seconds);

#if defined(__GNUC__) || defined(__clang__)
#define attribute_packed __attribute__((packed))
#define begin_packed
#define end_packed attribute_packed
#else
#define begin_packed __pragma( pack(push, 1) )
#define end_packed __pragma( pack(pop) )
#define attribute_packed !!! use begin_packed/end_packed instead !!!
#endif // usage: typedef begin_packed struct foo_s { ... } end_packed foo_t;

#ifndef byte_t
    #define byte_t uint8_t
#endif

#ifndef fp32_t
    #define fp32_t float
#endif

#ifndef fp64_t
    #define fp64_t double
#endif

#define thread_local __declspec(thread)

#define println(...) printline(__FILE__, __LINE__, __func__, "" __VA_ARGS__)

#define assertion(b, ...) do {                                              \
    if (!(b)) {                                                             \
        println("%s false\n", #b); println("" __VA_ARGS__);                 \
        printf("%s false\n", #b); printf("" __VA_ARGS__); printf("\n");     \
        __debugbreak();                                                     \
        exit(1);                                                            \
    }                                                                       \
} while (0) // better assert

#undef  assert
#define assert(...) assertion(__VA_ARGS__)

#define static_assertion(b) static_assert(b, #b)

#include "fp16.h"

enum {
    NSEC_IN_USEC = 1000,
    NSEC_IN_MSEC = NSEC_IN_USEC * 1000,
    NSEC_IN_SEC  = NSEC_IN_MSEC * 1000,
    MSEC_IN_SEC  = 1000,
    USEC_IN_SEC  = MSEC_IN_SEC * 1000
};

#define fatal_if(b, ...) do {                                    \
    bool _b_ = (b);                                              \
    if (_b_) {                                                   \
        printline(__FILE__, __LINE__, __func__, "%s", #b);       \
        printline(__FILE__, __LINE__, __func__, "" __VA_ARGS__); \
        fprintf(stderr, "%s(%d) %s() %s failed ",                \
                __FILE__, __LINE__, __func__, #b);               \
        fprintf(stderr, "" __VA_ARGS__);                         \
        __debugbreak();                                          \
        exit(1);                                                 \
    }                                                            \
} while (0)

#ifdef RT_IMPLEMENTATION

// or posix: long random(void);
// https://pubs.opengroup.org/onlinepubs/9699919799/functions/random.html

uint32_t random32(uint32_t* state) {
    // https://gist.github.com/tommyettinger/46a874533244883189143505d203312c
    static thread_local int init; // first seed must be odd
    if (init == 0) { init = 1; *state |= 1; }
    uint32_t z = (*state += 0x6D2B79F5UL);
    z = (z ^ (z >> 15)) * (z | 1UL);
    z ^= z + (z ^ (z >> 7)) * (z | 61UL);
    return z ^ (z >> 14);
}

#pragma comment(lib, "kernel32")

#define std_input_handle    ((uint32_t)-10)
#define std_output_handle   ((uint32_t)-11)
#define std_error_handle    ((uint32_t)-12)

typedef union LARGE_INTEGER {
    struct {
        uint32_t LowPart;
        int32_t HighPart;
    };
    int64_t QuadPart;
} LARGE_INTEGER;

void*    __stdcall GetStdHandle(uint32_t stdhandle);
int32_t  __stdcall QueryPerformanceCounter(LARGE_INTEGER* lpPerformanceCount);
int32_t  __stdcall QueryPerformanceFrequency(LARGE_INTEGER* lpFrequency);
void     __stdcall OutputDebugStringA(const char* s);
void*    __stdcall FindResourceA(void* module, const char* name, const char* type);
uint32_t __stdcall SizeofResource(void* module, void* res);
void*    __stdcall LoadResource(void* module, void* res);
void*    __stdcall LockResource(void* res);
void*    __stdcall LoadLibraryA(const char* pathname);
void*    __stdcall GetProcAddress(void* module, const char* pathname);


double seconds() { // since_boot
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
    static double one_over_freq;
    if (one_over_freq == 0) {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        one_over_freq = 1.0 / frequency.QuadPart;
    }
    return (double)qpc.QuadPart * one_over_freq;
}

int64_t nanoseconds() {
    return (int64_t)(seconds() * NSEC_IN_SEC);
}

/* posix:
uint64_t time_in_nanoseconds_absolute() {
    struct timespec tm = {0};
    clock_gettime(CLOCK_MONOTONIC, &tm);
    return NSEC_IN_SEC * (int64_t)tm.tv_sec + tm.tv_nsec;
}
*/

void printline(const char* file, int line, const char* func,
        const char* format, ...) {
    static thread_local char text[32 * 1024];
    va_list vl;
    va_start(vl, format);
    char* p = text + snprintf(text, countof(text), "%s(%d): %s() ",
        file, line, func);
    vsnprintf(p, countof(text) - (p - text), format, vl);
    text[countof(text) - 1] = 0;
    text[countof(text) - 2] = 0;
    size_t n = strlen(text);
    if (n > 0 && text[n - 1] != '\n') { text[n] = '\n'; text[n + 1] = 0;  }
    va_end(vl);
    OutputDebugStringA(text);
    // chop off full path from filename:
    p = text + strlen(file);
    while (p > text && *p != '\\' && *p != '/') { p--; }
    if (p != text) { p++; }
    fprintf(stderr, "%s", p);
}

int memmap_resource(const char* label, void* *data, int64_t *bytes) {
    enum { RT_RCDATA = 10 };
    void* res = FindResourceA(null, label, (const char*)RT_RCDATA);
    if (res != null) { *bytes = SizeofResource(null, res); }
    void* g = res != null ? LoadResource(null, res) : null;
    *data = g != null ? LockResource(g) : null;
    return *data != null ? 0 : 1;
}

// posix
// https://pubs.opengroup.org/onlinepubs/009695399/functions/dlsym.html
// https://pubs.opengroup.org/onlinepubs/009695399/functions/dlopen.html

void* load_dl(const char* pathname) {
    return LoadLibraryA(pathname); // dlopen() on Linux
}

void* find_symbol(void* dl, const char* symbol) {
    return GetProcAddress(dl, symbol); // dlsym on Linux
}

void sleep(double seconds) {
    assert(seconds >= 0);
    if (seconds < 0) { seconds = 0; }
    int64_t ns100 = (int64_t)(seconds * 1.0e+7); // in 0.1 us aka 100ns
    typedef int (__stdcall *nt_delay_execution_t)(int alertable,
        LARGE_INTEGER* delay_interval);
    static nt_delay_execution_t NtDelayExecution;
    // delay in 100-ns units. negative value means delay relative to current.
    LARGE_INTEGER delay = {0}; // delay in 100-ns units.
    delay.QuadPart = -ns100; // negative value means delay relative to current.
    if (NtDelayExecution == null) {
        static void* ntdll;
        if (ntdll == null) { ntdll = load_dl("ntdll.dll"); }
        fatal_if(ntdll == null);
        NtDelayExecution = (nt_delay_execution_t)find_symbol(ntdll,
            "NtDelayExecution");
        fatal_if(NtDelayExecution == null);
    }
    //  If "alertable": false is set to true, wait state can break in
    //  a result of NtAlertThread call.
    NtDelayExecution(false, &delay);
}

/* POSIX:
#include <time.h>
void sleep(double seconds) {
    struct timespec req = {
       .tv_sec  = (uint64_t)seconds;
       .tv_nsec = (uint64_t)(seconds * NSEC_IN_SEC) % NSEC_IN_SEC
    };
    nanosleep(&req, null);
}
*/

#endif // RT_IMPLEMENTATION

#ifdef __cplusplus
} // extern "C"
#endif
