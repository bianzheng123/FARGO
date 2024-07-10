//
// Created by 13172 on 2024/7/10.
//

// -*- c++ -*-

#include <cstdio>
#include <cassert>
#include <immintrin.h>
#include <string>

#include "distances_simd_avx512.h"

#ifdef __AVX2__
namespace faiss {

// reads 0 <= d < 4 floats as __m128
    static inline __m128 masked_read(int d, const float *x) {
        assert (0 <= d && d < 4);
        __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
        switch (d) {
            case 3:
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
        }
        return _mm_load_ps(buf);
        // cannot use AVX2 _mm_mask_set1_epi32
    }

    uint8_t lookup8bit[256];
//extern uint8_t lookup8bit[256];

    float
    fvec_inner_product_avx512(const float *x, const float *y, size_t d) {
        float res = 0.0f;
        for (int i = 0; i < d; i++) {
            res += x[i] * y[i];
        }
        return res;
    }

} // namespace faiss

#endif

