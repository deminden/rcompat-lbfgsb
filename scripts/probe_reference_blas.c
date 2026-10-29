/* Independent black-box calls to the public BLAS DDOT interface.
 * Compile with the reference environment's compiler and link its BLAS library.
 * No optimizer or BLAS implementation source is read or included. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern double ddot_(const int *, const double *, const int *, const double *, const int *);
static uint64_t state;
static double sample(void) {
    state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return (double)((int64_t)(state >> 32) - INT64_C(2147483648)) / (double)((state & 1023) + 1);
}
int main(void) {
    const int lengths[] = {0,1,2,3,4,9,10,11,13,16,31,32,33,63,64,65,96,127,128,129,255,256,257,512,1024};
    const uint64_t seeds[] = {1,2,3,7,19,123,256,1024};
    const int stride = 1;
    puts("length,seed,result_bits");
    for (unsigned j = 0; j < sizeof(lengths)/sizeof(*lengths); ++j) {
        int length = lengths[j];
        for (unsigned k = 0; k < sizeof(seeds)/sizeof(*seeds); ++k) {
            double x[1024], y[1024]; state = seeds[k];
            for (int i = 0; i < length; ++i) { x[i] = sample(); y[i] = sample(); }
            double actual = ddot_(&length, x, &stride, y, &stride);
            uint64_t bits; memcpy(&bits, &actual, sizeof(bits));
            printf("%d,%" PRIu64 ",%016" PRIx64 "\n", length, seeds[k], bits);
        }
    }
    return 0;
}
