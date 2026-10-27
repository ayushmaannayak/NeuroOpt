// Minimal hardware timing harness: copy generated code into an executable page,
// call it repeatedly and return the minimum elapsed TSC ticks. Faults inside the
// generated code are caught and reported instead of crashing the process.
#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <x86intrin.h>

static uint8_t *page = 0;
static const size_t PAGE_BYTES = 8u << 20;
static size_t loaded = 0;
static sigjmp_buf jb;
static volatile sig_atomic_t armed = 0;

static void on_signal(int sig) {
    if (armed) siglongjmp(jb, sig);
    signal(sig, SIG_DFL);
    raise(sig);
}

int nt_init(void) {
    if (page) return 0;
    page = mmap(0, PAGE_BYTES, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) { page = 0; return -1; }
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sa.sa_flags = SA_NODEFER;
    int sigs[] = {SIGSEGV, SIGILL, SIGFPE, SIGBUS, SIGTRAP};
    for (unsigned i = 0; i < sizeof sigs / sizeof *sigs; i++) sigaction(sigs[i], &sa, 0);
    return 0;
}

int nt_load(const uint8_t *code, size_t len) {
    if (!page || len > PAGE_BYTES) return -1;
    memcpy(page, code, len);
    loaded = len;
    return 0;
}

// Returns 0 on success, or the signal number if the code faulted.
int nt_time(int warmup, int reps, uint64_t *best_out) {
    void (*fn)(void) = (void (*)(void))page;
    int sig = sigsetjmp(jb, 1);
    if (sig) { armed = 0; return sig; }
    armed = 1;
    for (int i = 0; i < warmup; i++) fn();
    uint64_t best = ~0ull;
    unsigned aux;
    for (int r = 0; r < reps; r++) {
        _mm_lfence();
        uint64_t t0 = __rdtsc();
        _mm_lfence();
        fn();
        uint64_t t1 = __rdtscp(&aux);
        _mm_lfence();
        if (t1 - t0 < best) best = t1 - t0;
    }
    armed = 0;
    *best_out = best;
    return 0;
}
