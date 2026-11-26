//! Hardware-accelerated and portable SIMD matrix multiplication kernels for Kalman filter covariance updates.
//!
//! Provides optimized matrix multiplication ($A \cdot B$), matrix-transpose multiplication ($A \cdot B^T$),
//! and Joseph-stabilized covariance updates targeting:
//! - **x86_64**: AVX2 + FMA (`_mm256_fmadd_pd` / `_mm256_fmadd_ps`) with runtime CPU feature detection.
//! - **AArch64**: ARM NEON (`vfmaq_f64` / `vfmaq_f32`) vector pipelines.
//! - **Fallback**: Cache-friendly contiguous vector accumulation with loop unrolling for `#![no_std]` and all architectures.

use crate::common::Real;

/// Computes general matrix multiplication $C = A \cdot B$ where:
/// - $A$ is an $M \times K$ matrix
/// - $B$ is a $K \times N$ matrix
/// - $C$ is an $M \times N$ matrix (cleared and overwritten)
#[inline]
pub fn matmul<const M: usize, const K: usize, const N: usize>(
    a: &[[Real; K]; M],
    b: &[[Real; N]; K],
    out: &mut [[Real; N]; M],
) {
    // Zero out output buffer
    for row in out.iter_mut() {
        row.fill(0.0);
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified AVX2 and FMA support on x86_64 CPU at runtime.
            unsafe {
                matmul_avx2::<M, K, N>(a, b, out);
                return;
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(feature = "std"),
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: Compile-time target features guarantee AVX2 and FMA support.
        unsafe {
            matmul_avx2::<M, K, N>(a, b, out);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ARM NEON is standard baseline on AArch64 (ARMv8-A+).
        unsafe {
            matmul_neon::<M, K, N>(a, b, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    matmul_scalar::<M, K, N>(a, b, out);
}

/// Computes matrix multiplication with transposed second operand $C = A \cdot B^T$ where:
/// - $A$ is an $M \times K$ matrix
/// - $B$ is an $N \times K$ matrix (so $B^T$ is $K \times N$)
/// - $C$ is an $M \times N$ matrix (cleared and overwritten)
///
/// This operation is ubiquitous in Kalman covariance updates: $P_{next} = (FP) \cdot F^T$,
/// $P = (EP) \cdot E^T + (KR) \cdot K^T$, and $S = (HP) \cdot H^T + R$.
/// Because rows of both $A$ and $B$ are contiguous in row-major layout, this is computed
/// as vector dot products $C[i][j] = \langle A[i, :], B[j, :] \rangle$.
#[inline]
pub fn matmul_transpose_b<const M: usize, const K: usize, const N: usize>(
    a: &[[Real; K]; M],
    b: &[[Real; K]; N],
    out: &mut [[Real; N]; M],
) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified AVX2 and FMA support at runtime.
            unsafe {
                matmul_transpose_b_avx2::<M, K, N>(a, b, out);
                return;
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(feature = "std"),
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: Compile-time target features guarantee AVX2 and FMA.
        unsafe {
            matmul_transpose_b_avx2::<M, K, N>(a, b, out);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ARM NEON baseline on aarch64.
        unsafe {
            matmul_transpose_b_neon::<M, K, N>(a, b, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    matmul_transpose_b_scalar::<M, K, N>(a, b, out);
}

/// High-performance Joseph-stabilized covariance update:
/// $$P_{new} = (I - KH) P (I - KH)^T + K R K^T$$
///
/// Evaluated in $O(S^3 + S^2 M + S M^2)$ operations using SIMD kernels and guaranteed
/// numerical symmetry preservation.
pub fn joseph_form_covariance_update<const S: usize, const M: usize>(
    e: &[[Real; S]; S],
    p: &[[Real; S]; S],
    k: &[[Real; M]; S],
    r: &[[Real; M]; M],
    out: &mut [[Real; S]; S],
) {
    // 1. EP = E * P  (S x S)
    let mut ep = [[0.0; S]; S];
    matmul::<S, S, S>(e, p, &mut ep);

    // 2. KR = K * R  (S x M)
    let mut kr = [[0.0; M]; S];
    matmul::<S, M, M>(k, r, &mut kr);

    // 3. ep_et = (EP) * E^T (S x S)
    let mut ep_et = [[0.0; S]; S];
    matmul_transpose_b::<S, S, S>(&ep, e, &mut ep_et);

    // 4. kr_kt = (KR) * K^T (S x S)
    let mut kr_kt = [[0.0; S]; S];
    matmul_transpose_b::<S, M, S>(&kr, k, &mut kr_kt);

    // 5. Combine and enforce strict numerical symmetry
    for i in 0..S {
        out[i][i] = ep_et[i][i] + kr_kt[i][i];
        for j in 0..i {
            let avg = 0.5 * ((ep_et[i][j] + kr_kt[i][j]) + (ep_et[j][i] + kr_kt[j][i]));
            out[i][j] = avg;
            out[j][i] = avg;
        }
    }
}

/// Covariance propagation for Error-State Kalman Filters and general discrete linear systems:
/// $$P_{next} = F P F^T + Q_d$$
///
/// Followed by numerical symmetrization.
pub fn propagate_covariance<const N: usize>(
    f: &[[Real; N]; N],
    p: &[[Real; N]; N],
    q_diag: &[Real; N],
    out: &mut [[Real; N]; N],
) {
    // 1. FP = F * P
    let mut fp = [[0.0; N]; N];
    matmul::<N, N, N>(f, p, &mut fp);

    // 2. P_next = (FP) * F^T
    let mut next_p = [[0.0; N]; N];
    matmul_transpose_b::<N, N, N>(&fp, f, &mut next_p);

    // 3. Add diagonal process noise and enforce numerical symmetry
    for i in 0..N {
        out[i][i] = next_p[i][i] + q_diag[i];
        for j in 0..i {
            let avg = 0.5 * (next_p[i][j] + next_p[j][i]);
            out[i][j] = avg;
            out[j][i] = avg;
        }
    }
}

// ============================================================================
// Scalar Fallbacks (Portable, Cache-Friendly Row-Major Kernels)
// ============================================================================

#[inline]
pub fn matmul_scalar<const M: usize, const K: usize, const N: usize>(
    a: &[[Real; K]; M],
    b: &[[Real; N]; K],
    out: &mut [[Real; N]; M],
) {
    for i in 0..M {
        for k in 0..K {
            let a_ik = a[i][k];
            let b_row = &b[k];
            let out_row = &mut out[i];

            // 4-way unrolled inner loop across contiguous row entries
            let mut j = 0;
            while j + 3 < N {
                out_row[j] += a_ik * b_row[j];
                out_row[j + 1] += a_ik * b_row[j + 1];
                out_row[j + 2] += a_ik * b_row[j + 2];
                out_row[j + 3] += a_ik * b_row[j + 3];
                j += 4;
            }
            while j < N {
                out_row[j] += a_ik * b_row[j];
                j += 1;
            }
        }
    }
}

#[inline]
pub fn matmul_transpose_b_scalar<const M: usize, const K: usize, const N: usize>(
    a: &[[Real; K]; M],
    b: &[[Real; K]; N],
    out: &mut [[Real; N]; M],
) {
    for i in 0..M {
        let a_row = &a[i];
        for j in 0..N {
            let b_row = &b[j];
            let mut sum = 0.0;

            let mut k = 0;
            while k + 3 < K {
                sum += a_row[k] * b_row[k]
                    + a_row[k + 1] * b_row[k + 1]
                    + a_row[k + 2] * b_row[k + 2]
                    + a_row[k + 3] * b_row[k + 3];
                k += 4;
            }
            while k < K {
                sum += a_row[k] * b_row[k];
                k += 1;
            }
            out[i][j] = sum;
        }
    }
}

// ============================================================================
// x86_64 AVX2 + FMA Kernels
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_avx2<const M: usize, const K: usize, const N: usize>(
    a: &[[Real; K]; M],
    b: &[[Real; N]; K],
    out: &mut [[Real; N]; M],
) {
    #[cfg(not(feature = "single-precision"))]
    {
        use core::arch::x86_64::{
            _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd,
        };

        for i in 0..M {
            let out_ptr = out[i].as_mut_ptr();
            for k in 0..K {
                let a_val = a[i][k];
                let a_vec = _mm256_set1_pd(a_val);
                let b_ptr = b[k].as_ptr();

                let mut j = 0;
                while j + 3 < N {
                    let b_vec = _mm256_loadu_pd(b_ptr.add(j));
                    let c_vec = _mm256_loadu_pd(out_ptr.add(j));
                    let res = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                    _mm256_storeu_pd(out_ptr.add(j), res);
                    j += 4;
                }
                while j < N {
                    *out_ptr.add(j) += a_val * *b_ptr.add(j);
                    j += 1;
                }
            }
        }
    }

    #[cfg(feature = "single-precision")]
    {
        use core::arch::x86_64::{
            _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
        };

        for i in 0..M {
            let out_ptr = out[i].as_mut_ptr();
            for k in 0..K {
                let a_val = a[i][k];
                let a_vec = _mm256_set1_ps(a_val);
                let b_ptr = b[k].as_ptr();

                let mut j = 0;
                while j + 7 < N {
                    let b_vec = _mm256_loadu_ps(b_ptr.add(j));
                    let c_vec = _mm256_loadu_ps(out_ptr.add(j));
                    let res = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    _mm256_storeu_ps(out_ptr.add(j), res);
                    j += 8;
                }
                while j < N {
                    *out_ptr.add(j) += a_val * *b_ptr.add(j);
                    j += 1;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_transpose_b_avx2<const M: usize, const K: usize, const N: usize>(
    a: &[[Real; K]; M],
    b: &[[Real; K]; N],
    out: &mut [[Real; N]; M],
) {
    #[cfg(not(feature = "single-precision"))]
    {
        use core::arch::x86_64::{
            _mm256_extractf128_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm_add_pd,
        };

        for i in 0..M {
            let a_ptr = a[i].as_ptr();
            for j in 0..N {
                let b_ptr = b[j].as_ptr();
                let mut acc = _mm256_setzero_pd();

                let mut k = 0;
                while k + 3 < K {
                    let a_vec = _mm256_loadu_pd(a_ptr.add(k));
                    let b_vec = _mm256_loadu_pd(b_ptr.add(k));
                    acc = _mm256_fmadd_pd(a_vec, b_vec, acc);
                    k += 4;
                }

                // Horizontal reduction of __m256d
                let lo = _mm256_extractf128_pd(acc, 0);
                let hi = _mm256_extractf128_pd(acc, 1);
                let sum128 = _mm_add_pd(lo, hi);
                let mut tmp = [0.0; 2];
                core::arch::x86_64::_mm_storeu_pd(tmp.as_mut_ptr(), sum128);
                let mut sum = tmp[0] + tmp[1];

                while k < K {
                    sum += *a_ptr.add(k) * *b_ptr.add(k);
                    k += 1;
                }
                out[i][j] = sum;
            }
        }
    }

    #[cfg(feature = "single-precision")]
    {
        use core::arch::x86_64::{
            _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps,
        };

        for i in 0..M {
            let a_ptr = a[i].as_ptr();
            for j in 0..N {
                let b_ptr = b[j].as_ptr();
                let mut acc = _mm256_setzero_ps();

                let mut k = 0;
                while k + 7 < K {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(k));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(k));
                    acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                    k += 8;
                }

                // Horizontal reduction of __m256
                let lo = _mm256_extractf128_ps(acc, 0);
                let hi = _mm256_extractf128_ps(acc, 1);
                let sum128 = _mm_add_ps(lo, hi);
                let mut tmp = [0.0f32; 4];
                core::arch::x86_64::_mm_storeu_ps(tmp.as_mut_ptr(), sum128);
                let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

                while k < K {
                    sum += *a_ptr.add(k) * *b_ptr.add(k);
                    k += 1;
                }
                out[i][j] = sum;
            }
        }
    }
}

// ============================================================================
// ARM AArch64 NEON Kernels
// ============================================================================

#[cfg(target_arch = "aarch64")]
unsafe fn matmul_neon<const M: usize, const K: usize, const N: usize>(
    a: &[[Real; K]; M],
    b: &[[Real; N]; K],
    out: &mut [[Real; N]; M],
) {
    #[cfg(not(feature = "single-precision"))]
    {
        use core::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vst1q_f64};

        for i in 0..M {
            let out_ptr = out[i].as_mut_ptr();
            for k in 0..K {
                let a_val = a[i][k];
                let a_vec = vdupq_n_f64(a_val);
                let b_ptr = b[k].as_ptr();

                let mut j = 0;
                while j + 1 < N {
                    let b_vec = vld1q_f64(b_ptr.add(j));
                    let c_vec = vld1q_f64(out_ptr.add(j));
                    let res = vfmaq_f64(c_vec, a_vec, b_vec);
                    vst1q_f64(out_ptr.add(j), res);
                    j += 2;
                }
                while j < N {
                    *out_ptr.add(j) += a_val * *b_ptr.add(j);
                    j += 1;
                }
            }
        }
    }

    #[cfg(feature = "single-precision")]
    {
        use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

        for i in 0..M {
            let out_ptr = out[i].as_mut_ptr();
            for k in 0..K {
                let a_val = a[i][k];
                let a_vec = vdupq_n_f32(a_val);
                let b_ptr = b[k].as_ptr();

                let mut j = 0;
                while j + 3 < N {
                    let b_vec = vld1q_f32(b_ptr.add(j));
                    let c_vec = vld1q_f32(out_ptr.add(j));
                    let res = vfmaq_f32(c_vec, a_vec, b_vec);
                    vst1q_f32(out_ptr.add(j), res);
                    j += 4;
                }
                while j < N {
                    *out_ptr.add(j) += a_val * *b_ptr.add(j);
                    j += 1;
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn matmul_transpose_b_neon<const M: usize, const K: usize, const N: usize>(
    a: &[[Real; K]; M],
    b: &[[Real; K]; N],
    out: &mut [[Real; N]; M],
) {
    #[cfg(not(feature = "single-precision"))]
    {
        use core::arch::aarch64::{vaddvq_f64, vfmaq_f64, vld1q_f64};

        for i in 0..M {
            let a_ptr = a[i].as_ptr();
            for j in 0..N {
                let b_ptr = b[j].as_ptr();
                let mut acc = core::arch::aarch64::vdupq_n_f64(0.0);

                let mut k = 0;
                while k + 1 < K {
                    let a_vec = vld1q_f64(a_ptr.add(k));
                    let b_vec = vld1q_f64(b_ptr.add(k));
                    acc = vfmaq_f64(acc, a_vec, b_vec);
                    k += 2;
                }

                let mut sum = vaddvq_f64(acc);
                while k < K {
                    sum += *a_ptr.add(k) * *b_ptr.add(k);
                    k += 1;
                }
                out[i][j] = sum;
            }
        }
    }

    #[cfg(feature = "single-precision")]
    {
        use core::arch::aarch64::{vaddvq_f32, vfmaq_f32, vld1q_f32};

        for i in 0..M {
            let a_ptr = a[i].as_ptr();
            for j in 0..N {
                let b_ptr = b[j].as_ptr();
                let mut acc = core::arch::aarch64::vdupq_n_f32(0.0);

                let mut k = 0;
                while k + 3 < K {
                    let a_vec = vld1q_f32(a_ptr.add(k));
                    let b_vec = vld1q_f32(b_ptr.add(k));
                    acc = vfmaq_f32(acc, a_vec, b_vec);
                    k += 4;
                }

                let mut sum = vaddvq_f32(acc);
                while k < K {
                    sum += *a_ptr.add(k) * *b_ptr.add(k);
                    k += 1;
                }
                out[i][j] = sum;
            }
        }
    }
}
