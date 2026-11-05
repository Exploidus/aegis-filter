#[cfg(test)]
mod ukf {
    use aegis_filter::{Real, UnscentedKalmanFilter2D};
    #[test]
    fn linear_observation_multi_step() {
        // UKF with h(x) = x[0] (linear) — deterministically tracks a constant signal.
        // Multiple updates allow covariance to shrink and state to lock onto the target.
        let mut ukf = UnscentedKalmanFilter2D::new(
            [0.0, 0.0],
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.001, 0.0], [0.0, 0.001]],
            0.5,
        )
        .unwrap();

        let target = 5.0;
        for _ in 0..30 {
            ukf.predict(0.1, |x, _dt| [x[0], x[1]]).unwrap();
            let _ = ukf.update(target, |x| x[0], None).unwrap();
        }

        // After 30 updates the position estimate must be close to target
        assert!((ukf.x[0] - target).abs() < 0.3, "ukf.x[0] = {}", ukf.x[0]);
    }

    #[test]
    fn nonlinear_tight_covariance() {
        // h(x) = x[0]^2 — non-linear. With tight initial covariance sigma
        // points stay close to mean, so linearisation error is small and a
        // single update step visibly moves the state toward the true value.
        let true_pos: Real = 3.0;
        let mut ukf = UnscentedKalmanFilter2D::new(
            [true_pos + 0.5, 0.0],
            [[0.04, 0.0], [0.0, 0.04]], // tight — sigma points ±0.2 from mean
            [[0.001, 0.0], [0.0, 0.001]],
            0.1,
        )
        .unwrap();

        ukf.predict(0.0, |x, _dt| [x[0], x[1]]).unwrap();

        let z = true_pos * true_pos; // 9.0
        let before = ukf.x[0];
        let res = ukf.update(z, |x| x[0] * x[0], None).unwrap();

        // Update must move x[0] toward true_pos (3.0) from initial 3.5
        assert!(
            res[0] < before,
            "Expected x[0] to decrease toward 3.0, before={before}, after={}",
            res[0]
        );
        assert!(res[0] > 2.5, "Expected x[0] > 2.5, got {}", res[0]);
    }

    #[test]
    fn predict_update_characterization() {
        // Golden pins for the unscented propagation and update paths, guarding the
        // iterator refactor in issue #19. Captured from the current implementation;
        // reordering the weighted sums would shift these beyond 1e-12.
        let mut ukf = UnscentedKalmanFilter2D::new(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.01, 0.0], [0.0, 0.01]],
            0.5,
        )
        .unwrap();

        #[cfg(feature = "single-precision")]
        let tol: Real = 1e-4;
        #[cfg(not(feature = "single-precision"))]
        let tol: Real = 1e-12;

        ukf.predict(0.1, |x, _dt| [x[0] + 0.1, x[1]]).unwrap();
        let est = ukf.update(1.0, |x| x[0], None).unwrap();
        assert!(
            (est[0] - 0.7019867549668872).abs() < tol,
            "est[0] = {}",
            est[0]
        );
        assert!((est[1] - 0.0).abs() < tol, "est[1] = {}", est[1]);
        assert!(
            (ukf.p[0][0] - 0.3344370860927156).abs() < tol,
            "p00 = {}",
            ukf.p[0][0]
        );
        assert!(
            (ukf.p[1][1] - 1.0099999999999998).abs() < tol,
            "p11 = {}",
            ukf.p[1][1]
        );

        // A second step with a non-trivial transition exercises the covariance
        // propagation (x1 drift) and the innovation / cross-covariance sums.
        ukf.predict(0.1, |x, _dt| [x[0] + 0.1, x[1] + 0.05])
            .unwrap();
        let est2 = ukf.update(1.2, |x| x[0], None).unwrap();
        assert!(
            (est2[0] - 0.9643322092384911).abs() < tol,
            "est2[0] = {}",
            est2[0]
        );
        assert!(
            (est2[1] - 0.050000000000000044).abs() < tol,
            "est2[1] = {}",
            est2[1]
        );
    }
}

#[cfg(test)]
mod eskf {
    use aegis_filter::{
        ErrorStateKalmanFilter, EskfNoiseParameters, EskfNominalState, FilterError, Quaternion,
        Vector3,
    };

    #[test]
    fn stationary_kinematics() {
        let init_state = EskfNominalState::new(
            Vector3::new(0.0, 0.0, 100.0),
            Vector3::ZERO,
            Quaternion::IDENTITY,
            Vector3::ZERO,
            Vector3::ZERO,
        );
        let cov_diag = [0.1; 15];
        let noise = EskfNoiseParameters::new(0.01, 0.001, 0.0001, 0.00001);
        let mut eskf = ErrorStateKalmanFilter::new(
            init_state,
            cov_diag,
            noise,
            Vector3::new(0.0, 0.0, -9.80665),
        )
        .unwrap();

        // Stationary IMU on Earth surface: accelerometer reads +1g upward reaction
        let accel = Vector3::new(0.0, 0.0, 9.80665);
        let gyro = Vector3::ZERO;
        let dt = 0.01;

        for _ in 0..100 {
            eskf.predict(&accel, &gyro, dt).unwrap();
        }

        // Stationary drone should remain at altitude 100m
        assert!((eskf.state().position.z - 100.0).abs() < 1e-4);
        assert!(eskf.state().velocity.norm() < 1e-4);
        assert!((eskf.state().attitude.w - 1.0).abs() < 1e-5);
    }

    #[test]
    fn position_update_and_outlier_gating() {
        let init_state = EskfNominalState::new(
            Vector3::new(10.0, 5.0, -2.0),
            Vector3::ZERO,
            Quaternion::IDENTITY,
            Vector3::ZERO,
            Vector3::ZERO,
        );
        let cov_diag = [1.0; 15];
        let noise = EskfNoiseParameters::new(0.05, 0.005, 0.0001, 0.00001);
        let mut eskf = ErrorStateKalmanFilter::new(
            init_state,
            cov_diag,
            noise,
            Vector3::new(0.0, 0.0, -9.80665),
        )
        .unwrap();

        // Legitimate GNSS position observation (sub-meter drift)
        let gnss_pos = Vector3::new(10.2, 5.1, -1.9);
        eskf.update_position(&gnss_pos, 0.25, None).unwrap();

        assert!((eskf.state().position.x - 10.2).abs() < 0.2);
        assert!((eskf.state().position.y - 5.1).abs() < 0.2);

        // Adversarial GNSS spoofing / multipath spike (10,000 meters away)
        let spoofed_pos = Vector3::new(10000.0, 0.0, 0.0);
        let err = eskf.update_position(&spoofed_pos, 0.25, Some(3.0));
        assert_eq!(err, Err(FilterError::MeasurementGated));
    }
}

#[cfg(test)]
mod fusion {
    use aegis_filter::{
        FilterError, FusionStatus, MultiSensorFusion1D, SensorMeasurement, SensorSource,
    };
    #[test]
    fn consecutive_anomaly_reset() {
        let mut fusion = MultiSensorFusion1D::<2>::new(0.0, 1.0, 0.01).unwrap();
        // Sensor requires 2 consecutive anomalies to quarantine
        let sensor = SensorSource::new(1, 0.2, 2).unwrap();
        fusion.register_sensor(sensor).unwrap();

        // 1. First anomaly -> Gated
        let st1 = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 100.0,
            })
            .unwrap();
        assert_eq!(st1, FusionStatus::Gated);

        // 2. Interleaved healthy measurement -> Fused (breaks streak, resets anomaly_count)
        let st2 = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 0.01,
            })
            .unwrap();
        assert_eq!(st2, FusionStatus::Fused);

        // 3. Second anomaly (isolated, not consecutive) -> Gated, NOT Quarantined
        let st3 = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 100.0,
            })
            .unwrap();
        assert_eq!(st3, FusionStatus::Gated);
        assert!((fusion.sensor_health(1).unwrap() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn singularity_guard() {
        let mut fusion = MultiSensorFusion1D::<1>::new(0.0, 1.0, 0.01).unwrap();
        let sensor = SensorSource::new(1, 0.5, 3).unwrap();
        fusion.register_sensor(sensor).unwrap();

        // Force degenerate/singular variance
        fusion.p = -0.5;
        let err = fusion.fuse(SensorMeasurement {
            sensor_id: 1,
            value: 1.0,
        });
        assert_eq!(err, Err(FilterError::SingularInnovation));
    }
}
