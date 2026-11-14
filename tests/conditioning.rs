#![allow(clippy::needless_range_loop)]

use aegis_filter::{
    check_matrix_conditioning, matrix_condition_number, matrix_norm_1, matrix_norm_frobenius,
    matrix_norm_inf, ErrorStateKalmanFilter, EskfNoiseParameters, EskfNominalState, FilterError,
    LinearKalmanFilter, Quaternion, Real, UnscentedKalmanFilter2D, Vector3,
    DEFAULT_MAX_CONDITION_NUMBER,
};

#[test]
fn test_matrix_norms() {
    // 2x2 matrix: [[1.0, -2.0], [3.0, 4.0]]
    // Column 0: |1| + |3| = 4.0
    // Column 1: |-2| + |4| = 6.0
    // 1-norm (max col sum) = 6.0
    // Row 0: |1| + |-2| = 3.0
    // Row 1: |3| + |4| = 7.0
    // Inf-norm (max row sum) = 7.0
    // Frobenius norm: sqrt(1^2 + 4 + 9 + 16) = sqrt(30) ≈ 5.477225575
    let m = [[1.0, -2.0], [3.0, 4.0]];

    let norm1 = matrix_norm_1(&m);
    assert!((norm1 - 6.0).abs() < 1e-12);

    let norm_inf = matrix_norm_inf(&m);
    assert!((norm_inf - 7.0).abs() < 1e-12);

    let norm_frob = matrix_norm_frobenius(&m);
    assert!((norm_frob - (30.0f64).sqrt() as Real).abs() < 1e-10);
}

#[test]
fn test_identity_and_scaled_diagonal_condition_number() {
    // Identity matrix has condition number exactly 1.0
    let eye3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let cond = matrix_condition_number(&eye3).unwrap();
    assert!((cond - 1.0).abs() < 1e-12);

    // Diagonal matrix with scale differences
    // diag([100.0, 1.0, 0.01]) -> cond = 100.0 / 0.01 = 10000.0
    let diag = [[100.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.01]];
    let cond_diag = matrix_condition_number(&diag).unwrap();
    assert!((cond_diag - 10000.0).abs() < 1e-6);

    // Well-conditioned validation against default threshold
    assert!(check_matrix_conditioning(&diag, DEFAULT_MAX_CONDITION_NUMBER).is_ok());
    // Should fail if threshold is tighter than 10000
    assert_eq!(
        check_matrix_conditioning(&diag, 5000.0),
        Err(FilterError::InvalidCovariance)
    );
}

#[test]
fn test_hilbert_matrix_condition_numbers() {
    // Hilbert matrix 3x3:
    // [ 1.0,      1/2,      1/3     ]
    // [ 1/2,      1/3,      1/4     ]
    // [ 1/3,      1/4,      1/5     ]
    // Theoretical 1-norm condition number for H_3 is approx 748.0
    let h3 = [
        [1.0, 1.0 / 2.0, 1.0 / 3.0],
        [1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0],
        [1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0],
    ];
    let cond3 = matrix_condition_number(&h3).unwrap();
    assert!(
        (cond3 - 748.0).abs() < 1.0,
        "Expected kappa_1(H3) ≈ 748, got {cond3}"
    );

    // Hilbert matrix 4x4:
    // Theoretical 1-norm condition number for H_4 is approx 28375
    let mut h4 = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            h4[i][j] = 1.0 / ((i + j + 1) as Real);
        }
    }
    let cond4 = matrix_condition_number(&h4).unwrap();
    assert!(
        (cond4 - 28375.0).abs() < 100.0,
        "Expected kappa_1(H4) ≈ 28375, got {cond4}"
    );

    // Hilbert matrix 5x5:
    // Theoretical 1-norm condition number for H_5 is approx 9.43656e5 (943656)
    let mut h5 = [[0.0; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            h5[i][j] = 1.0 / ((i + j + 1) as Real);
        }
    }
    let cond5 = matrix_condition_number(&h5).unwrap();
    assert!(
        (cond5 - 943656.0).abs() < 100.0,
        "Expected kappa_1(H5) ≈ 943656, got {cond5}"
    );
}

#[test]
fn test_ill_conditioned_covariance_rejection() {
    // Severely ill-conditioned matrix with sensor noise scale discrepancy:
    // Variance ratio is 10^14, which exceeds standard double-precision threshold (10^12)
    #[cfg(not(feature = "single-precision"))]
    let extreme_cov = [[1e8, 0.0], [0.0, 1e-6]];

    #[cfg(feature = "single-precision")]
    let extreme_cov = [[1e4, 0.0], [0.0, 1e-3]];

    let cond = matrix_condition_number(&extreme_cov).unwrap();
    assert!(cond >= 1e7);

    // Must be rejected by check_matrix_conditioning
    let res = check_matrix_conditioning(&extreme_cov, 1e6);
    assert_eq!(res, Err(FilterError::InvalidCovariance));
}

#[test]
fn test_singular_and_non_finite_matrices() {
    // Singular rank-deficient matrix (collinear rows)
    let singular = [[1.0, 2.0], [2.0, 4.0]];
    let res = matrix_condition_number(&singular);
    assert_eq!(res, Err(FilterError::NumericalInstability));

    // Matrix with NaN / Inf
    let nan_mat = [[f64::NAN as Real, 0.0], [0.0, 1.0]];
    assert_eq!(
        matrix_condition_number(&nan_mat),
        Err(FilterError::NonFiniteInput)
    );

    // Non-symmetric covariance matrix must be rejected by check_matrix_conditioning
    let asymmetric = [[1.0, 0.5], [0.1, 1.0]];
    assert_eq!(
        check_matrix_conditioning(&asymmetric, 1e6),
        Err(FilterError::InvalidCovariance)
    );

    // Negative diagonal variance must be rejected
    let neg_var = [[-1.0, 0.0], [0.0, 1.0]];
    assert_eq!(
        check_matrix_conditioning(&neg_var, 1e6),
        Err(FilterError::InvalidCovariance)
    );
}

#[test]
fn test_linear_kalman_filter_conditioning_methods() {
    // 2-state linear filter
    let f = LinearKalmanFilter::<2, 1>::new(
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 2.0]],
        [[1.0, 0.1], [0.0, 1.0]],
        [[0.01, 0.0], [0.0, 0.01]],
        [[1.0, 0.0]],
        [[0.5]],
    )
    .unwrap();

    let cond = f.condition_number().unwrap();
    assert!((cond - 2.0).abs() < 1e-6);
    assert!(f.is_well_conditioned());
    assert!(f.check_conditioning(10.0).is_ok());
    assert_eq!(
        f.check_conditioning(1.5),
        Err(FilterError::InvalidCovariance)
    );
}

#[test]
fn test_unscented_kalman_filter_conditioning_methods() {
    let ukf = UnscentedKalmanFilter2D::new(
        [0.0, 0.0],
        [[5.0, 0.0], [0.0, 0.5]],
        [[0.01, 0.0], [0.0, 0.01]],
        0.1,
    )
    .unwrap();

    let cond = ukf.condition_number().unwrap();
    // diag([5.0, 0.5]) -> cond = 10.0
    assert!((cond - 10.0).abs() < 1e-6);
    assert!(ukf.is_well_conditioned());
}

#[test]
fn test_eskf_covariance_conditioning_methods() {
    let nominal = EskfNominalState {
        position: Vector3::new(0.0, 0.0, 0.0),
        velocity: Vector3::new(0.0, 0.0, 0.0),
        attitude: Quaternion::new(1.0, 0.0, 0.0, 0.0),
        accel_bias: Vector3::new(0.0, 0.0, 0.0),
        gyro_bias: Vector3::new(0.0, 0.0, 0.0),
    };
    let mut diag_cov = [1.0; 15];
    diag_cov[0] = 100.0;
    diag_cov[14] = 0.1;

    let noise = EskfNoiseParameters {
        accel_noise: 0.1,
        gyro_noise: 0.01,
        accel_bias_noise: 0.001,
        gyro_bias_noise: 0.0001,
    };
    let gravity = Vector3::new(0.0, 0.0, -9.80665);

    let eskf = ErrorStateKalmanFilter::new(nominal, diag_cov, noise, gravity).unwrap();

    let cond = eskf.condition_number().unwrap();
    // max = 100, min = 0.1 -> cond = 1000.0
    assert!((cond - 1000.0).abs() < 1e-3);
    assert!(eskf.is_well_conditioned());
    assert!(eskf.check_conditioning(5000.0).is_ok());
}
