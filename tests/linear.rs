#[cfg(test)]
mod linear {
    use aegis_filter::{
        common::{FilterError, Real},
        linear::*,
        DEFAULT_OUTLIER_GATE_SIGMA,
    };

    // (1) predict regression: a constant-velocity model must still propagate x.
    // F = [[1, 0.1], [0, 1]] -> position advances by 0.1*velocity per step; with
    // velocity held at 100, x[0] grows by 10.0 each predict. Deterministic.
    #[test]
    fn predict_kinematic() {
        let mut f = LinearKalmanFilter::<2, 1>::new(
            [0.0, 100.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.1], [0.0, 1.0]],
            [[0.001, 0.0], [0.0, 0.001]],
            [[1.0, 0.0]], // H: one row, position only
            [[0.5]],
        )
        .unwrap();
        for _ in 0..3 {
            f.predict().unwrap();
        }
        assert!((f.x[0] - 30.0).abs() < 1e-9, "x[0] = {}", f.x[0]);
    }

    #[cfg(test)]
    mod update {
        use super::*;
        // (2) M=1 measurement update converges the state to a constant target, exactly
        // like the 1D scalar filter. S=1, M=1 with F=I: x tracks the target as the
        // posterior variance shrinks.
        #[test]
        fn update_m1_convergence() {
            let mut f = LinearKalmanFilter::<1, 1>::new(
                [0.0],
                [[10.0]],
                [[1.0]],
                [[0.01]],
                [[1.0]],
                [[0.5]],
            )
            .unwrap();
            let target = 25.0;
            for _ in 0..50 {
                f.predict().unwrap();
                let _ = f.update(&[target], None).unwrap();
            }
            assert!((f.x[0] - target).abs() < 0.2, "x[0] = {}", f.x[0]);
        }

        // (3) The H=I, P=I, R=I case has an exact closed-form first update: K = P H^T
        // S^-1 = I . I . (2I)^-1 = 0.5 I, so x moves halfway to the measurement in one step.
        #[test]
        fn update_m2_single_step_halfway() {
            let mut f = LinearKalmanFilter::<2, 2>::new(
                [0.0, 0.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]], // F = I
                [[0.0, 0.0], [0.0, 0.0]], // Q = 0
                [[1.0, 0.0], [0.0, 1.0]], // H = I
                [[1.0, 0.0], [0.0, 1.0]], // R = I
            )
            .unwrap();
            let z = [10.0, 20.0];
            let out = f.update(&z, None).unwrap();
            assert!((out[0] - 5.0).abs() < 1e-9, "out[0] = {}", out[0]);
            assert!((out[1] - 10.0).abs() < 1e-9, "out[1] = {}", out[1]);
        }

        // (4) Repeated M=2 updates converge to the true measurement vector.
        #[test]
        fn update_m2_convergence() {
            let mut f = LinearKalmanFilter::<2, 2>::new(
                [0.0, 0.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.01, 0.0], [0.0, 0.01]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            )
            .unwrap();
            let z = [10.0, 20.0];
            for _ in 0..50 {
                f.predict().unwrap();
                f.update(&z, None).unwrap();
            }
            assert!((f.x[0] - 10.0).abs() < 0.1, "x[0] = {}", f.x[0]);
            assert!((f.x[1] - 20.0).abs() < 0.1, "x[1] = {}", f.x[1]);
        }

        // (5) Joseph form must keep P symmetric and positive on the diagonals. For the
        // H=I, P=I, R=I case the posterior covariance is the analytic 0.50 * I:
        // P = (I - KH) P (I - KH)^T + K R K^T = 0.25 I + 0.25 I = 0.50 I.
        #[test]
        fn update_joseph_symmetry_and_psd() {
            let mut f = LinearKalmanFilter::<2, 2>::new(
                [0.0, 0.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            )
            .unwrap();
            f.update(&[5.0, 5.0], None).unwrap();
            // Symmetry: off-diagonals match on both triangles.
            assert!((f.p[0][1] - f.p[1][0]).abs() < 1e-9, "off-diag mismatch");
            assert!((f.p[0][1]).abs() < 1e-9, "off-diag = {}", f.p[0][1]);
            // Diagonals: analytic 0.50, and strictly positive (PSD lower bound).
            assert!((f.p[0][0] - 0.50).abs() < 1e-9, "p00 = {}", f.p[0][0]);
            assert!((f.p[1][1] - 0.50).abs() < 1e-9, "p11 = {}", f.p[1][1]);
        }

        // (6) Gating: a far-off M=1 measurement is rejected (3-sigma gate) and the state
        // is left untouched, mirroring the scalar filter's MeasurementGated behaviour.
        #[test]
        fn update_m1_gating_rejects_outlier() {
            let mut f = LinearKalmanFilter::<1, 1>::new(
                [0.0],
                [[1.0]],
                [[1.0]],
                [[0.0]],
                [[1.0]],
                [[0.01]],
            )
            .unwrap();
            let err = f.update(&[1000.0], Some(3.0));
            assert_eq!(err, Err(FilterError::MeasurementGated));
            assert!((f.x[0] - 0.0).abs() < 1e-12, "state moved on gated update");
        }

        // (7) A rank-deficient observation (both rows identical) with zero process and
        // measurement noise makes the innovation covariance singular -> NumericalInstability.
        #[test]
        fn update_singular_innovation() {
            let mut f = LinearKalmanFilter::<2, 2>::new(
                [0.0, 0.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0]], // rank-deficient H
                [[0.0, 0.0], [0.0, 0.0]], // R = 0, so S = H P H^T is singular
            )
            .unwrap();
            assert_eq!(
                f.update(&[1.0, 1.0], None).err(),
                Some(FilterError::NumericalInstability)
            );
        }

        // (8) Non-finite measurements are rejected at the boundary, like the 1D path.
        #[test]
        fn update_nonfinite_measurement() {
            let mut f = LinearKalmanFilter::<1, 1>::new(
                [0.0],
                [[1.0]],
                [[1.0]],
                [[0.01]],
                [[1.0]],
                [[0.5]],
            )
            .unwrap();
            assert_eq!(
                f.update(&[Real::NAN], None).err(),
                Some(FilterError::NonFiniteInput)
            );
            assert_eq!(
                f.update(&[Real::INFINITY], None).err(),
                Some(FilterError::NonFiniteInput)
            );
        }

        // (9) Non-diagonal H with correlated (non-diagonal) measurement error must use the true
        // K = P H^T S^-1, not a column-sum of S^-1. With P=I, H=[[1,1],[0,1]], R=I the
        // innovation covariance is S = H H^T + I = [[3,1],[1,2]], inverse [[2/5,-1/5],[-1/5,3/5]],
        // so K = H^T S^-1 = [[0.4,-0.2],[0.2,0.4]]; with x=0, x_new = K * z.
        // z=[4,2] is chosen so the off-diagonal K[0][1] (correct -0.2, old bug 0) is exposed:
        // the old index bug gave x_new[0] = 0.8 instead of 1.2.
        #[test]
        fn update_k_non_diagonal_correlated() {
            let mut f = LinearKalmanFilter::<2, 2>::new(
                [0.0, 0.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [0.0, 1.0]], // non-diagonal H
                [[1.0, 0.0], [0.0, 1.0]],
            )
            .unwrap();
            let out = f.update(&[4.0, 2.0], None).unwrap();
            #[cfg(feature = "single-precision")]
            let tol: Real = 1e-4;
            #[cfg(not(feature = "single-precision"))]
            let tol: Real = 1e-9;
            assert!((out[0] - 1.2).abs() < tol, "out[0] = {}", out[0]);
            assert!((out[1] - 1.6).abs() < tol, "out[1] = {}", out[1]);
        }
        // (9) Factored O(S^3) Joseph update test for S=4, M=2: verifies that the factored
        // (E*P)*E^T + (K*S)*K^T produces identical positive semi-definite, symmetric covariance.
        #[test]
        #[allow(clippy::needless_range_loop)]
        fn update_higher_dim_factored_joseph() {
            let x0 = [10.0, 0.5, 20.0, -0.2];
            let p0 = [
                [2.0, 0.1, 0.0, 0.0],
                [0.1, 1.5, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.1],
                [0.0, 0.0, 0.1, 1.5],
            ];
            let f_mat = [
                [1.0, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let q_mat = [
                [0.01, 0.0, 0.0, 0.0],
                [0.0, 0.01, 0.0, 0.0],
                [0.0, 0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01, 0.0],
            ];
            // Measure position x and position y
            let h_mat = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
            let r_mat = [[0.25, 0.02], [0.02, 0.25]];

            let mut kf =
                LinearKalmanFilter::<4, 2>::new(x0, p0, f_mat, q_mat, h_mat, r_mat).unwrap();
            kf.predict().unwrap();
            let z = [10.15, 19.95];
            let updated_state = kf.update(&z, Some(DEFAULT_OUTLIER_GATE_SIGMA)).unwrap();

            // State moves toward measurement
            assert!((updated_state[0] - 10.15).abs() < 1.0);
            assert!((updated_state[2] - 19.95).abs() < 1.0);

            // Verify covariance symmetry P[i][j] == P[j][i]
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        (kf.p[i][j] - kf.p[j][i]).abs() < 1e-12,
                        "Covariance asymmetry at ({}, {}): {} vs {}",
                        i,
                        j,
                        kf.p[i][j],
                        kf.p[j][i]
                    );
                }
                // Strict positive variance on diagonal
                assert!(kf.p[i][i] > 0.0, "Diagonal variance non-positive at {}", i);
                assert!(
                    kf.p[i][i] < p0[i][i] + 0.1,
                    "Variance did not contract on measurement update"
                );
            }
        }
    }

    // (10) Fail-closed security validation: when initialized with new_resilient or with_default_gate,
    // calls to update(..., None) automatically enforce 3-sigma innovation gating.
    #[test]
    fn fail_closed_gating_behavior() {
        let mut f_scalar = ScalarKalmanFilter::new_resilient(0.0, 1.0, 0.01, 0.5).unwrap();
        // 1000-sigma outlier passed with None: MUST be rejected by default 3-sigma gate!
        let err = f_scalar.update(1000.0, None);
        assert_eq!(err, Err(FilterError::MeasurementGated));
        assert!((f_scalar.state() - 0.0).abs() < 1e-12);

        // Explicit opt-out via Some(-1.0) allows through
        let accepted = f_scalar.update(1.0, Some(-1.0)).unwrap();
        assert!(accepted > 0.0);

        // Same for LinearKalmanFilter
        let mut f_linear = LinearKalmanFilter::<1, 1>::new_resilient(
            [0.0],
            [[1.0]],
            [[1.0]],
            [[0.01]],
            [[1.0]],
            [[0.5]],
        )
        .unwrap();
        let err_linear = f_linear.update(&[1000.0], None);
        assert_eq!(err_linear, Err(FilterError::MeasurementGated));
    }
}
