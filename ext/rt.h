#pragma once
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#if _MSC_VER
#include <io.h>
#include <malloc.h>
#else
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif
#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <memory.h>

// runtime to mitigate tiny diffs of Linux, MacOS and Windows

#if __clang__
// https://clang.llvm.org/docs/DiagnosticsReference.html
// #pragma clang diagnostic ignored "-Wunused-variable"
// #pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wformat"
#pragma clang diagnostic ignored "-Wformat-invalid-specifier"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef countof
    #define countof(a) (sizeof(a) / sizeof((a)[0])) // MS is _countof()
#endif

#define minimum(a, b) ((a) < (b) ? (a) : (b))
#define maximum(a, b) ((a) > (b) ? (a) : (b))

#define null ((void*)0) // like nullptr but for C99

uint32_t random32(uint32_t* state); // state aka seed
double   seconds(void);     // seconds since boot (3/10MHz resolution)
uint64_t nanoseconds(void); // nanoseconds since boot (3/10MHz resolution)
void     printline(const char* file, int line, const char* func,
                   const char* format, ...);
int      memmap_resource(const char* label, void* *data, int64_t *bytes);
void*    load_dl(const char* pathname); // dlopen | LoadLibrary
void*    find_symbol(void* dl, const char* symbol); // dlsym | GetProcAddress
int      mem_map(const char* filename, void** data, int64_t* bytes, bool ro);
void     mem_unmap(void* address, int64_t bytes);
void     sleep_for(double seconds);
void     _debugbreak(void); // breakpoint

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

#if _MSC_VER
#define thread_local __declspec(thread)
#else
#define thread_local __thread
#endif

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

#if _MSC_VER

#pragma comment(lib, "kernel32")

#define std_input_handle    ((uint32_t)-10)
#define std_output_handle   ((uint32_t)-11)
#define std_error_handle    ((uint32_t)-12)

void*    __stdcall GetStdHandle(uint32_t stdhandle);
int32_t  __stdcall QueryPerformanceCounter(int64_t* performance_count);
int32_t  __stdcall QueryPerformanceFrequency(int64_t* frequency);
int32_t  __stdcall GetFileSizeEx(void* file, int64_t* size);
void     __stdcall OutputDebugStringA(const char* s);
void*    __stdcall FindResourceA(void* module, const char* name, const char* type);
uint32_t __stdcall SizeofResource(void* module, void* res);
void*    __stdcall LoadResource(void* module, void* res);
void*    __stdcall LockResource(void* res);
void*    __stdcall LoadLibraryA(const char* pathname);
void*    __stdcall GetProcAddress(void* module, const char* pathname);
int32_t  __stdcall GetLastError(void);
int32_t  __stdcall CloseHandle(void* handle);
void*    __stdcall CreateFileMappingA(void* file, void* attributes, uint32_t protect,
                    uint32_t maximum_size_high, uint32_t maximum_size_low,
                    const char* name);
void*    __stdcall MapViewOfFile(void* mapping, uint32_t access,
                    uint32_t file_offset_high, uint32_t file_offset_low,
                    size_t number_of_bytes_to_map);
void*    __stdcall CreateFileA(const char* filename, uint32_t access,
                    uint32_t share_mode, void* security_attributes,
                    uint32_t creation_disposition,
                    uint32_t flags_and_attributes, void* template_file);
int32_t  __stdcall UnmapViewOfFile(void* address);

double seconds() { // since_boot
    int64_t qpc = 0;
    QueryPerformanceCounter(&qpc);
    static double one_over_freq;
    if (one_over_freq == 0) {
        int64_t frequency = 0;
        QueryPerformanceFrequency(&frequency);
        one_over_freq = 1.0 / frequency;
    }
    return (double)qpc * one_over_freq;
}

int64_t nanoseconds() {
    return (int64_t)(seconds() * NSEC_IN_SEC);
}

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

void sleep_for(double seconds) {
    assert(seconds >= 0);
    if (seconds < 0) { seconds = 0; }
    int64_t ns100 = (int64_t)(seconds * 1.0e+7); // in 0.1 us aka 100ns
    typedef int (__stdcall *nt_delay_execution_t)(int alertable, int64_t* delay_interval);
    static nt_delay_execution_t NtDelayExecution;
    // delay in 100-ns units. negative value means delay relative to current.
    int64_t delay = {0}; // delay in 100-ns units.
    delay = -ns100; // negative value means delay relative to current.
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

int mem_map_file(void* file, void** data, int64_t* bytes, bool ro) {
    int r = 0;
    void* address = null;
    int64_t size = 0;
    enum {
        PAGE_READONLY  = 0x02,
        PAGE_READWRITE = 0x04
    };
    enum {
        FILE_MAP_READ  = 0x02,
        FILE_MAP_WRITE = 0x04
    };
    if (GetFileSizeEx(file, &size)) {
        void* map_file = CreateFileMappingA(file, NULL,
            ro ? PAGE_READONLY : PAGE_READWRITE, 0, size, null);
        if (map_file == null) {
            r = GetLastError();
        } else {
            address = MapViewOfFile(map_file, ro ?
                FILE_MAP_READ : FILE_MAP_READ|FILE_MAP_WRITE, 0, 0, size);
            if (address != null) {
                *bytes = size;
            } else {
                r = GetLastError();
            }
            int b = CloseHandle(map_file); // not setting errno because CloseHandle is expected to work here
            assert(b); (void)b;
        }
    } else {
        r = GetLastError();
    }
    if (r == 0) { *data = address; }
    return r;
}

int mem_map(const char* filename, void** data, int64_t* bytes, bool ro) {
    *bytes = 0; // important for empty files - which result in (null, 0) and errno == 0
    int r = 0;
    enum {
        GENERIC_READ   = 0x80000000L,
        GENERIC_WRITE  = 0x40000000L
    };
    enum {
        FILE_SHARE_READ   = 0x00000001,
        FILE_SHARE_WRITE  = 0x00000002,
        FILE_SHARE_DELETE = 0x00000004
    };
    enum {
        CREATE_NEW        = 1,
        CREATE_ALWAYS     = 2,
        OPEN_EXISTING     = 3,
        OPEN_ALWAYS       = 4,
        TRUNCATE_EXISTING = 5,
        FILE_ATTRIBUTE_NORMAL = 0x00000080
    };
    #define INVALID_HANDLE_VALUE ((void*)(intptr_t)-1)
    uint32_t access = ro ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
    // w/o FILE_SHARE_DELETE RAM-based files: FILE_FLAG_DELETE_ON_CLOSE | FILE_ATTRIBUTE_TEMPORARY won't open
    uint32_t share  = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    void* file = CreateFileA(filename, access, share, null, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, null);
    if (file == INVALID_HANDLE_VALUE) {
        r = GetLastError();
    } else {
        r = mem_map_file(file, data, bytes, ro);
        int b = CloseHandle(file); // not setting errno because CloseHandle is expected to work here
        assert(b); (void)b;
    }
    return r;
}

void mem_unmap(void* address, int64_t bytes) {
    (void)bytes; /* bytes unused, need by posix version */
    if (address != null) {
        fatal_if(!UnmapViewOfFile(address));
    }
}

#else // POSIX and APPLE

void _debugbreak(void) {
    raise(SIGTRAP);
}

uint64_t nanoseconds(void) {
    struct timespec tm = {0};
    clock_gettime(CLOCK_MONOTONIC, &tm);
    return NSEC_IN_SEC * (int64_t)tm.tv_sec + tm.tv_nsec;
}

double seconds(void) {
    return (double)nanoseconds() / NSEC_IN_SEC;
}

void sleep_for(double seconds) {
    struct timespec req = {
       .tv_sec  = (uint64_t)seconds,
       .tv_nsec = (uint64_t)(seconds * NSEC_IN_SEC) % NSEC_IN_SEC
    };
    nanosleep(&req, null);
}

int mem_map(const char* filename, void** data, int64_t* bytes, bool ro) {
    int r = 0;
    int fd = open(filename, ro ? O_RDONLY : O_RDWR);
    if (fd >= 0) {
        int length = (int)lseek(fd, 0, SEEK_END);
        if (0 < length && length <= 0x7FFFFFFF) {
            *data = mmap(0, length, ro ? PROT_READ : PROT_READ|PROT_WRITE, ro ? MAP_PRIVATE : MAP_SHARED, fd, 0);
            if (*data != null) { *bytes = length; }
            else { r = errno; }
        }
        close(fd);
    } else { r = errno; }
    return r;
}

void mem_unmap(void* address, int64_t bytes) {
    if (address != null) {
        munmap(address, bytes);
    }
}

void printline(const char* file, int line, const char* func,
        const char* format, ...) {
    // chop off full path from filename:
    const char* p = file + strlen(file) - 1;
    while (p > file && *p != '/') { p--; }
    if (p != file) { p++; }
    va_list vl;
    va_start(vl, format);
    printf("%s(%d): %s() ", p, line, func);
    vprintf(format, vl);
    printf("\n");
}

#endif

#endif // RT_IMPLEMENTATION

#ifdef __cplusplus
} // extern "C"
#endif

