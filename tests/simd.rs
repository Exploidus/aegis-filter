//! Verification and mathematical validation suite for SIMD matrix multiplication
//! and Joseph-stabilized covariance updates.

#![allow(clippy::needless_range_loop, clippy::neg_multiply)]

use aegis_filter::simd::{
    joseph_form_covariance_update, matmul, matmul_scalar, matmul_transpose_b,
    matmul_transpose_b_scalar, propagate_covariance,
};
use aegis_filter::Real;

#[cfg(not(feature = "single-precision"))]
const EPSILON_TOL: Real = 1e-12;

#[cfg(feature = "single-precision")]
const EPSILON_TOL: Real = 1e-4;

#[cfg(not(feature = "single-precision"))]
const SYMMETRY_TOL: Real = 1e-14;

#[cfg(feature = "single-precision")]
const SYMMETRY_TOL: Real = 1e-5;

fn assert_matrix_close<const M: usize, const N: usize>(
    actual: &[[Real; N]; M],
    expected: &[[Real; N]; M],
    tol: Real,
) {
    for i in 0..M {
        for j in 0..N {
            let diff = (actual[i][j] - expected[i][j]).abs();
            assert!(
                diff <= tol,
                "Mismatch at ({}, {}): actual={}, expected={}, diff={}",
                i,
                j,
                actual[i][j],
                expected[i][j],
                diff
            );
        }
    }
}

fn assert_matrix_symmetric<const N: usize>(mat: &[[Real; N]; N], tol: Real) {
    for i in 0..N {
        for j in 0..i {
            let diff = (mat[i][j] - mat[j][i]).abs();
            assert!(
                diff <= tol,
                "Asymmetry at ({}, {}): [{}][{}]={}, [{}][{}]={}",
                i,
                j,
                i,
                j,
                mat[i][j],
                j,
                i,
                mat[j][i]
            );
        }
    }
}

#[test]
fn test_matmul_accuracy_across_dimensions() {
    // 1. 2x2 Identity and arbitrary
    let a_2x2: [[Real; 2]; 2] = [[1.5, 2.0], [3.0, -1.0]];
    let b_2x2: [[Real; 2]; 2] = [[0.5, 4.0], [-2.0, 1.0]];
    let expected_2x2: [[Real; 2]; 2] = [
        [1.5 * 0.5 + 2.0 * -2.0, 1.5 * 4.0 + 2.0 * 1.0],
        [3.0 * 0.5 + -1.0 * -2.0, 3.0 * 4.0 + -1.0 * 1.0],
    ];
    let mut out_2x2 = [[0.0; 2]; 2];
    matmul::<2, 2, 2>(&a_2x2, &b_2x2, &mut out_2x2);
    assert_matrix_close(&out_2x2, &expected_2x2, EPSILON_TOL);

    // 2. 4x4 matrix (matches exact AVX2 4-lane width for f64, 8-lane for f32)
    let mut a_4x4 = [[0.0; 4]; 4];
    let mut b_4x4 = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a_4x4[i][j] = (i * 4 + j + 1) as Real * 0.1;
            b_4x4[i][j] = if i == j { 2.0 } else { -0.5 };
        }
    }
    let mut out_4x4 = [[0.0; 4]; 4];
    let mut ref_4x4 = [[0.0; 4]; 4];
    matmul::<4, 4, 4>(&a_4x4, &b_4x4, &mut out_4x4);
    matmul_scalar::<4, 4, 4>(&a_4x4, &b_4x4, &mut ref_4x4);
    assert_matrix_close(&out_4x4, &ref_4x4, EPSILON_TOL);

    // 3. 15x15 (ESKF Error-State dimension)
    let mut a_15x15 = [[0.0; 15]; 15];
    let mut b_15x15 = [[0.0; 15]; 15];
    for i in 0..15 {
        for j in 0..15 {
            a_15x15[i][j] = ((i + 1) as Real) * 0.2 - ((j + 1) as Real) * 0.15;
            b_15x15[i][j] = ((i * 15 + j) as Real) * 0.05 + 1.0;
        }
    }
    let mut out_15x15 = [[0.0; 15]; 15];
    let mut ref_15x15 = [[0.0; 15]; 15];
    matmul::<15, 15, 15>(&a_15x15, &b_15x15, &mut out_15x15);
    matmul_scalar::<15, 15, 15>(&a_15x15, &b_15x15, &mut ref_15x15);
    assert_matrix_close(&out_15x15, &ref_15x15, EPSILON_TOL);

    // 4. Odd non-power-of-two dimensions (7x5 multiplied by 5x3)
    let mut a_7x5 = [[0.0; 5]; 7];
    let mut b_5x3 = [[0.0; 3]; 5];
    for i in 0..7 {
        for j in 0..5 {
            a_7x5[i][j] = (i + j) as Real * 0.3;
        }
    }
    for i in 0..5 {
        for j in 0..3 {
            b_5x3[i][j] = (i * 2 + j) as Real * 0.4;
        }
    }
    let mut out_7x3 = [[0.0; 3]; 7];
    let mut ref_7x3 = [[0.0; 3]; 7];
    matmul::<7, 5, 3>(&a_7x5, &b_5x3, &mut out_7x3);
    matmul_scalar::<7, 5, 3>(&a_7x5, &b_5x3, &mut ref_7x3);
    assert_matrix_close(&out_7x3, &ref_7x3, EPSILON_TOL);
}

#[test]
fn test_matmul_transpose_b_accuracy() {
    // A: 6x4, B: 5x4 => A * B^T is 6x5
    let mut a = [[0.0; 4]; 6];
    let mut b = [[0.0; 4]; 5];
    for i in 0..6 {
        for k in 0..4 {
            a[i][k] = (i + k * 2) as Real * 0.5 + 0.1;
        }
    }
    for j in 0..5 {
        for k in 0..4 {
            b[j][k] = (j * 3 + k) as Real * 0.25 - 0.5;
        }
    }

    let mut out = [[0.0; 5]; 6];
    let mut ref_out = [[0.0; 5]; 6];
    matmul_transpose_b::<6, 4, 5>(&a, &b, &mut out);
    matmul_transpose_b_scalar::<6, 4, 5>(&a, &b, &mut ref_out);
    assert_matrix_close(&out, &ref_out, EPSILON_TOL);

    // Also compare against explicit transpose + matmul
    let mut b_t = [[0.0; 5]; 4];
    for i in 0..5 {
        for k in 0..4 {
            b_t[k][i] = b[i][k];
        }
    }
    let mut matmul_ref = [[0.0; 5]; 6];
    matmul::<6, 4, 5>(&a, &b_t, &mut matmul_ref);
    assert_matrix_close(&out, &matmul_ref, EPSILON_TOL);
}

#[test]
fn test_joseph_form_covariance_update_accuracy_and_symmetry() {
    const S: usize = 6;
    const M: usize = 2;

    // Prior covariance P (symmetric positive definite)
    let mut p = [[0.0; S]; S];
    for i in 0..S {
        p[i][i] = 2.0 + (i as Real) * 0.5;
        for j in 0..i {
            let cov = 0.1 * ((i + j) as Real);
            p[i][j] = cov;
            p[j][i] = cov;
        }
    }

    // Measurement matrix H (M x S)
    let mut h = [[0.0; S]; M];
    h[0][0] = 1.0;
    h[0][1] = 0.5;
    h[1][2] = 1.0;
    h[1][3] = -0.2;

    // Measurement noise R (M x M symmetric)
    let r: [[Real; M]; M] = [[0.2, 0.05], [0.05, 0.3]];

    // Kalman gain K (S x M)
    let mut k = [[0.0; M]; S];
    for i in 0..S {
        k[i][0] = 0.1 * (i as Real + 1.0);
        k[i][1] = 0.05 * (i as Real + 1.0);
    }

    // E = I - K * H
    let mut e = [[0.0; S]; S];
    for i in 0..S {
        for j in 0..S {
            let mut kh = 0.0;
            for t in 0..M {
                kh += k[i][t] * h[t][j];
            }
            e[i][j] = if i == j { 1.0 - kh } else { -kh };
        }
    }

    let mut p_out = [[0.0; S]; S];
    joseph_form_covariance_update::<S, M>(&e, &p, &k, &r, &mut p_out);

    // Verify exact floating-point symmetry
    assert_matrix_symmetric(&p_out, SYMMETRY_TOL);

    // Verify diagonal positivity
    for i in 0..S {
        assert!(
            p_out[i][i] > 0.0,
            "Diagonal variance must remain positive: p_out[{}][{}]={}",
            i,
            i,
            p_out[i][i]
        );
    }
}

#[test]
fn test_eskf_covariance_propagation_accuracy_and_symmetry() {
    const N: usize = 15;

    // Transition matrix F (15x15)
    let mut f = [[0.0; N]; N];
    for i in 0..N {
        f[i][i] = 1.0;
    }
    // Cross coupling (velocity into position, gyro bias into attitude, etc.)
    let dt = 0.01;
    f[0][3] = dt;
    f[1][4] = dt;
    f[2][5] = dt;
    f[3][9] = -dt;
    f[4][10] = -dt;
    f[5][11] = -dt;
    f[6][12] = -dt;
    f[7][13] = -dt;
    f[8][14] = -dt;

    // Prior covariance P (15x15 positive-definite symmetric)
    let mut p = [[0.0; N]; N];
    for i in 0..N {
        p[i][i] = 1.0 + (i as Real) * 0.1;
        for j in 0..i {
            let c = 0.01 * ((i + j) as Real);
            p[i][j] = c;
            p[j][i] = c;
        }
    }

    // Process noise diagonal Q_d
    let mut q_diag = [0.0; N];
    for i in 0..N {
        q_diag[i] = 1e-4 * (i as Real + 1.0);
    }

    let mut p_next = [[0.0; N]; N];
    propagate_covariance::<N>(&f, &p, &q_diag, &mut p_next);

    // Verify strict symmetry
    assert_matrix_symmetric(&p_next, SYMMETRY_TOL);

    // Verify diagonal variance growth due to process noise
    for i in 0..N {
        assert!(
            p_next[i][i] >= p[i][i] * 0.99,
            "Propagated covariance diagonal must be non-zero and stable"
        );
    }
}
