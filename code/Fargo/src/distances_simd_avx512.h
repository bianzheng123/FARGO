//
// Created by 13172 on 2024/7/10.
//

#ifndef FARGO_DISTANCES_SIMD_AVX512_H
#define FARGO_DISTANCES_SIMD_AVX512_H

#include <stddef.h>
#include <stdint.h>

#ifdef __AVX2__
namespace faiss {

/*********************************************************
 * Optimized distance/norm/inner prod computations
 *********************************************************/

/// inner product
    float fvec_inner_product_avx512(
            const float *x,
            const float *y,
            size_t d);

} // namespace faiss


#endif
#endif //FARGO_DISTANCES_SIMD_AVX512_H
