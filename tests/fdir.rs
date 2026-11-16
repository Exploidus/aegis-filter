//! Integration and stress tests for Fault Detection, Isolation, and Recovery (FDIR)
//! and Chi-square / Mahalanobis outlier gating under extreme sensor dropouts.
//!
//! Addresses Issue #30:
//! - Complete IMU dropouts / zero-readings protecting 6-DOF odometry
//! - High-frequency oscillatory sensor spikes triggering quarantine
//! - Corrupted covariance inputs (non-positive definite measurement noise)
//! - Quarantined state transitions protecting core state estimation and multi-sensor handover

#[cfg(test)]
mod fdir_edge_cases {
    use aegis_filter::{
        ErrorStateKalmanFilter, EskfNoiseParameters, EskfNominalState, FilterError, FloatOps,
        FusionStatus, LinearKalmanFilter, MultiSensorFusion1D, Quaternion, Real,
        ScalarKalmanFilter, SensorMeasurement, SensorSource, Vector3,
    };

    #[cfg(feature = "single-precision")]
    const TOL: Real = 1e-3;
    #[cfg(not(feature = "single-precision"))]
    const TOL: Real = 1e-7;

    // =========================================================================
    // 1. Complete IMU Dropouts / Zero-Readings
    // =========================================================================

    #[test]
    fn test_imu_complete_dropout_zero_readings_protects_odometry() {
        // Drone in forward flight at 100m altitude, 10 m/s velocity along X
        let init_state = EskfNominalState::new(
            Vector3::new(0.0, 0.0, 100.0),
            Vector3::new(10.0, 0.0, 0.0),
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
        .expect("Failed to initialize ESKF");

        let dt = 0.01;

        // Step 1: Nominal stationary/constant-velocity flight with IMU reading gravity reaction
        let nominal_accel = Vector3::new(0.0, 0.0, 9.80665);
        let nominal_gyro = Vector3::ZERO;
        for _ in 0..20 {
            eskf.predict(&nominal_accel, &nominal_gyro, dt)
                .expect("Nominal prediction must succeed");
        }
        assert!((eskf.state().position.z - 100.0).abs_val() < 0.1);

        // Step 2: Complete IMU dropout / communication blackout
        // Sensor stream abruptly drops to zero-readings (e.g., bus disconnected, flatlined registers)
        let zero_accel = Vector3::ZERO;
        let zero_gyro = Vector3::ZERO;

        // Ingest 50 consecutive zero-readings (0.5s of total dropout)
        for _ in 0..50 {
            // Must not panic, divide by zero, or produce NaN
            eskf.predict(&zero_accel, &zero_gyro, dt)
                .expect("IMU zero-reading prediction must never crash");
        }

        // Verify that internal states remain strictly valid finite numbers
        assert!(eskf.state().position.is_valid());
        assert!(eskf.state().velocity.is_valid());
        assert!(eskf.state().attitude.norm().is_valid_float());
        assert!(
            eskf.is_well_conditioned(),
            "Covariance must remain well-conditioned after IMU dropouts"
        );

        // Step 3: During IMU dropout, a severe sensor anomaly arrives (e.g. GPS drop to 0,0,0)
        let corrupt_gps = Vector3::new(0.0, 0.0, 0.0);
        let gate_res = eskf.update_position(&corrupt_gps, 0.5, Some(3.0));
        assert_eq!(
            gate_res,
            Err(FilterError::MeasurementGated),
            "Corrupt GPS drop during IMU blackout must be rejected by Chi-square gate"
        );

        // Odometry altitude should not have collapsed to 0
        assert!(eskf.state().position.z > 50.0);

        // Step 4: Legitimate external GNSS fix arrives to re-anchor odometry
        let valid_pos = Vector3::new(
            eskf.state().position.x,
            eskf.state().position.y,
            eskf.state().position.z + 0.1,
        );
        eskf.update_position(&valid_pos, 0.2, Some(3.0))
            .expect("Valid GPS update after dropout must be fused cleanly");
    }

    #[test]
    fn test_fdir_sensor_zero_reading_dropout_and_quarantine_protection() {
        // Multi-sensor fusion engine with 3 redundant altimeters on an aircraft
        // Sensor 1: Primary GPS Altimeter (Variance R = 0.5, max 3 consecutive anomalies)
        // Sensor 2: Backup Barometer (Variance R = 0.2, max 5 consecutive anomalies)
        // Sensor 3: LiDAR Altimeter (Variance R = 0.05, max 3 consecutive anomalies)
        let mut fusion =
            MultiSensorFusion1D::<3>::new(200.0, 1.0, 0.01).expect("Failed to initialize fusion");

        let gps = SensorSource::new(1, 0.5, 3).unwrap().with_auto_recovery(3);
        let baro = SensorSource::new(2, 0.2, 5).unwrap();
        let lidar = SensorSource::new(3, 0.05, 3).unwrap();

        fusion.register_sensor(gps).unwrap();
        fusion.register_sensor(baro).unwrap();
        fusion.register_sensor(lidar).unwrap();

        // Nominal flight at altitude 200m
        for _ in 0..5 {
            fusion.predict(0.0).unwrap();
            assert_eq!(
                fusion
                    .fuse(SensorMeasurement {
                        sensor_id: 1,
                        value: 200.1
                    })
                    .unwrap(),
                FusionStatus::Fused
            );
            assert_eq!(
                fusion
                    .fuse(SensorMeasurement {
                        sensor_id: 2,
                        value: 199.95
                    })
                    .unwrap(),
                FusionStatus::Fused
            );
            assert_eq!(
                fusion
                    .fuse(SensorMeasurement {
                        sensor_id: 3,
                        value: 200.02
                    })
                    .unwrap(),
                FusionStatus::Fused
            );
        }

        let pre_failure_altitude = fusion.state();
        let pre_failure_variance = fusion.variance();

        // Catastrophic sensor 1 failure: complete dropout transmitting flatline 0.0m readings
        // Strike 1: Gated
        let st1 = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 0.0,
            })
            .unwrap();
        assert_eq!(st1, FusionStatus::Gated);
        assert_eq!(fusion.is_sensor_quarantined(1), Some(false));
        // State must remain uncorrupted
        assert!((fusion.state() - pre_failure_altitude).abs_val() < TOL);

        // Strike 2: Gated
        let st2 = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 0.0,
            })
            .unwrap();
        assert_eq!(st2, FusionStatus::Gated);

        // Strike 3: Quarantined!
        let st3 = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 0.0,
            })
            .unwrap();
        assert_eq!(st3, FusionStatus::Quarantined);
        assert_eq!(fusion.is_sensor_quarantined(1), Some(true));
        assert_eq!(fusion.sensor_health(1), Some(0.0));

        // Subsequent zero readings while quarantined are silently ignored
        for _ in 0..20 {
            let ignored = fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 0.0,
                })
                .unwrap();
            assert_eq!(ignored, FusionStatus::IgnoredQuarantined);
        }

        // Meanwhile, Barometer (Sensor 2) and LiDAR (Sensor 3) continue updating odometry
        for step in 1..=10 {
            let alt_true = 200.0 + (step as Real) * 0.5;
            fusion.predict(0.5).unwrap();

            let s2 = fusion
                .fuse(SensorMeasurement {
                    sensor_id: 2,
                    value: alt_true - 0.05,
                })
                .unwrap();
            let s3 = fusion
                .fuse(SensorMeasurement {
                    sensor_id: 3,
                    value: alt_true + 0.02,
                })
                .unwrap();
            assert_eq!(s2, FusionStatus::Fused);
            assert_eq!(s3, FusionStatus::Fused);

            // Quarantined GPS continues sending 0.0, still safely ignored
            let s1 = fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 0.0,
                })
                .unwrap();
            assert_eq!(s1, FusionStatus::IgnoredQuarantined);
        }

        // Verify core altitude tracked the true climb (205m) rather than the 0m GPS dropout
        assert!((fusion.state() - 205.0).abs_val() < 0.5);
        assert!(fusion.variance() < pre_failure_variance + 1.0);
    }

    // =========================================================================
    // 2. High-Frequency Oscillatory Sensor Spikes
    // =========================================================================

    #[test]
    fn test_high_frequency_oscillatory_sensor_spikes_quarantine() {
        let mut fusion = MultiSensorFusion1D::<2>::new(50.0, 1.0, 0.01).unwrap();
        // Sensor requires 2 consecutive anomalies to quarantine
        let sensor = SensorSource::new(1, 0.2, 2).unwrap();
        fusion.register_sensor(sensor).unwrap();

        // 1st spike (+1000m): Gated
        let st1 = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 1050.0,
            })
            .unwrap();
        assert_eq!(st1, FusionStatus::Gated);
        assert_eq!(fusion.sensor_health(1), Some(0.5));

        // 2nd rapid opposite spike (-1000m): Quarantined
        let st2 = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: -950.0,
            })
            .unwrap();
        assert_eq!(st2, FusionStatus::Quarantined);
        assert_eq!(fusion.sensor_health(1), Some(0.0));
        assert_eq!(fusion.is_sensor_quarantined(1), Some(true));

        // Ingest 500 rapid high-frequency alternating spikes (+500, -500, +1000, -1000)
        let state_before = fusion.state();
        let var_before = fusion.variance();

        for i in 0..500 {
            let spike = if i % 2 == 0 {
                50.0 + 500.0
            } else {
                50.0 - 500.0
            };
            let res = fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: spike,
                })
                .unwrap();
            assert_eq!(res, FusionStatus::IgnoredQuarantined);
        }

        // Assert zero odometry contamination
        assert_eq!(fusion.state(), state_before);
        assert_eq!(fusion.variance(), var_before);
    }

    #[test]
    fn test_oscillatory_chattering_near_gate_boundary_health_degradation() {
        let mut fusion = MultiSensorFusion1D::<1>::new(0.0, 1.0, 0.01).unwrap();
        // Sensor allows up to 5 consecutive anomalies
        let sensor = SensorSource::new(1, 0.2, 5).unwrap();
        fusion.register_sensor(sensor).unwrap();

        // Intermittent chattering: anomaly, clean, anomaly, clean...
        // Multiplicative decay (0.5x) vs additive recovery (+0.1) causes net degradation
        let mut health_scores = [0.0; 6];
        for item in &mut health_scores {
            // Anomaly: Gated
            let _ = fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 50.0,
                })
                .unwrap();
            // Clean: Fused
            let _ = fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 0.01,
                })
                .unwrap();
            *item = fusion.sensor_health(1).unwrap();
        }

        // Verify monotonic/net health degradation over chattering cycles
        assert!(
            health_scores[5] < health_scores[0],
            "Chattering must degrade health score"
        );
        assert!(health_scores[5] < 0.4);
    }

    #[test]
    fn test_multi_dim_high_frequency_spikes_mahalanobis_gating() {
        // 2D Tracking filter: state = [x, vx], measurement = [x]
        let dt: Real = 0.1;
        let f = [[1.0, dt], [0.0, 1.0]];
        let q = [[0.01, 0.0], [0.0, 0.01]];
        let h = [[1.0, 0.0]];
        let r = [[0.2]];

        let mut filter = LinearKalmanFilter::<2, 1>::new_resilient(
            [10.0, 1.0],
            [[1.0, 0.0], [0.0, 1.0]],
            f,
            q,
            h,
            r,
        )
        .expect("Failed to create resilient filter");

        let initial_x = filter.x;
        let initial_p = filter.p;

        // Rapid oscillatory vector spikes
        for step in 0..100 {
            let spike_val = if step % 2 == 0 { 500.0 } else { -500.0 };
            let res = filter.update(&[spike_val], None);
            assert_eq!(
                res,
                Err(FilterError::MeasurementGated),
                "Oscillatory spike must be gated by Mahalanobis gate"
            );
        }

        // State and covariance must remain strictly identical
        assert_eq!(filter.x, initial_x);
        assert_eq!(filter.p, initial_p);
    }

    // =========================================================================
    // 3. Corrupted Covariance Inputs (Non-Positive Definite Noise)
    // =========================================================================

    #[test]
    fn test_corrupted_covariance_inputs_non_positive_definite_noise() {
        // 1. Negative variance
        assert_eq!(
            SensorSource::new(1, -0.01, 3).err(),
            Some(FilterError::InvalidParameter)
        );
        assert_eq!(
            SensorSource::new(1, -100.0, 3).err(),
            Some(FilterError::InvalidParameter)
        );

        // 2. Zero variance
        assert_eq!(
            SensorSource::new(1, 0.0, 3).err(),
            Some(FilterError::InvalidParameter)
        );

        // 3. Non-finite variance
        assert_eq!(
            SensorSource::new(1, Real::NAN, 3).err(),
            Some(FilterError::InvalidParameter)
        );
        assert_eq!(
            SensorSource::new(1, Real::INFINITY, 3).err(),
            Some(FilterError::InvalidParameter)
        );
        assert_eq!(
            SensorSource::new(1, Real::NEG_INFINITY, 3).err(),
            Some(FilterError::InvalidParameter)
        );

        // 4. Zero/Negative Gate Threshold
        let valid_sensor = SensorSource::new(1, 0.5, 3).unwrap();
        assert_eq!(
            valid_sensor.with_gate_threshold(0.0).err(),
            Some(FilterError::InvalidParameter)
        );
        assert_eq!(
            valid_sensor.with_gate_threshold(-2.5).err(),
            Some(FilterError::InvalidParameter)
        );
        assert_eq!(
            valid_sensor.with_gate_threshold(Real::NAN).err(),
            Some(FilterError::InvalidParameter)
        );

        // 5. Corrupted variance in ScalarKalmanFilter
        assert_eq!(
            ScalarKalmanFilter::new(0.0, 1.0, 0.01, 0.0).err(),
            Some(FilterError::InvalidCovariance)
        );
        assert_eq!(
            ScalarKalmanFilter::new(0.0, 1.0, 0.01, -1.0).err(),
            Some(FilterError::InvalidCovariance)
        );
    }

    #[test]
    fn test_corrupted_noise_variance_in_eskf_odometry() {
        let init_state = EskfNominalState::new(
            Vector3::new(0.0, 0.0, 10.0),
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

        let pos = Vector3::new(0.0, 0.0, 10.0);
        let vel = Vector3::ZERO;

        // Position update: negative, zero, and non-finite variance rejected
        assert_eq!(
            eskf.update_position(&pos, -0.5, Some(3.0)),
            Err(FilterError::NonFiniteInput)
        );
        assert_eq!(
            eskf.update_position(&pos, 0.0, Some(3.0)),
            Err(FilterError::NonFiniteInput)
        );
        assert_eq!(
            eskf.update_position(&pos, Real::NAN, Some(3.0)),
            Err(FilterError::NonFiniteInput)
        );

        // Velocity update: negative and non-finite variance rejected
        assert_eq!(
            eskf.update_velocity(&vel, -0.1, Some(3.0)),
            Err(FilterError::NonFiniteInput)
        );
        assert_eq!(
            eskf.update_velocity(&vel, Real::INFINITY, Some(3.0)),
            Err(FilterError::NonFiniteInput)
        );

        // Altimeter update: negative variance rejected
        assert_eq!(
            eskf.update_altimeter(10.0, -1.0, Some(3.0)),
            Err(FilterError::NonFiniteInput)
        );

        // State remains undamaged
        assert!((eskf.state().position.z - 10.0).abs_val() < TOL);
    }

    #[test]
    fn test_non_positive_definite_multidim_noise_covariance() {
        // Multi-dimensional filter: R matrix with negative diagonal
        let f = [[1.0, 0.1], [0.0, 1.0]];
        let p = [[1.0, 0.0], [0.0, 1.0]];
        let q = [[0.01, 0.0], [0.0, 0.01]];
        let h = [[1.0, 0.0], [0.0, 1.0]];

        // Negative diagonal in R
        let r_corrupted = [[-0.5, 0.0], [0.0, 0.5]];
        let res = LinearKalmanFilter::<2, 2>::new([0.0, 0.0], p, f, q, h, r_corrupted);
        assert_eq!(
            res.err(),
            Some(FilterError::InvalidCovariance),
            "Negative diagonal covariance must be rejected at initialization"
        );
    }

    // =========================================================================
    // 4. Quarantined State Transitions & Handover Protection
    // =========================================================================

    #[test]
    fn test_fdir_full_lifecycle_and_auto_recovery_handover() {
        let mut fusion = MultiSensorFusion1D::<2>::new(100.0, 1.0, 0.01).unwrap();
        // Sensor 1: Auto-recovery after 3 consecutive clean measurements
        let s1 = SensorSource::new(1, 0.2, 2).unwrap().with_auto_recovery(3);
        // Sensor 2: Permanent backup
        let s2 = SensorSource::new(2, 0.5, 5).unwrap();

        fusion.register_sensor(s1).unwrap();
        fusion.register_sensor(s2).unwrap();

        // 1. Initial nominal fusion
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 100.05
                })
                .unwrap(),
            FusionStatus::Fused
        );

        // 2. Attack: 2 consecutive anomalies force quarantine
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 500.0
                })
                .unwrap(),
            FusionStatus::Gated
        );
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 500.0
                })
                .unwrap(),
            FusionStatus::Quarantined
        );
        assert_eq!(fusion.is_sensor_quarantined(1), Some(true));

        // 3. Sensor 1 attempts recovery, but has an intermittent glitch at sample 2
        // Clean sample 1
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 100.02
                })
                .unwrap(),
            FusionStatus::IgnoredQuarantined
        );
        // Glitch interrupts recovery streak -> resets consecutive_healthy
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 300.0
                })
                .unwrap(),
            FusionStatus::IgnoredQuarantined
        );
        assert_eq!(fusion.sensor(1).unwrap().consecutive_healthy, 0);

        // 4. Now 3 uninterrupted healthy samples arrive -> Recovered!
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 100.01
                })
                .unwrap(),
            FusionStatus::IgnoredQuarantined
        );
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 100.02
                })
                .unwrap(),
            FusionStatus::IgnoredQuarantined
        );
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 100.01
                })
                .unwrap(),
            FusionStatus::Recovered
        );
        assert_eq!(fusion.is_sensor_quarantined(1), Some(false));
        assert_eq!(fusion.sensor_health(1), Some(0.5));

        // 5. Subsequent sample is actively fused again
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 100.03
                })
                .unwrap(),
            FusionStatus::Fused
        );
        assert_eq!(fusion.sensor_health(1), Some(0.6));
    }

    #[test]
    fn test_all_sensors_quarantined_dead_reckoning_fallback() {
        let mut fusion = MultiSensorFusion1D::<2>::new(0.0, 1.0, 0.05).unwrap();
        let s1 = SensorSource::new(1, 0.2, 1).unwrap();
        let s2 = SensorSource::new(2, 0.3, 1).unwrap();

        fusion.register_sensor(s1).unwrap();
        fusion.register_sensor(s2).unwrap();

        // Single extreme anomaly quarantines both sensors
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 1,
                    value: 1000.0
                })
                .unwrap(),
            FusionStatus::Quarantined
        );
        assert_eq!(
            fusion
                .fuse(SensorMeasurement {
                    sensor_id: 2,
                    value: -1000.0
                })
                .unwrap(),
            FusionStatus::Quarantined
        );

        assert_eq!(fusion.is_sensor_quarantined(1), Some(true));
        assert_eq!(fusion.is_sensor_quarantined(2), Some(true));

        // With all sensors quarantined, filter dead-reckons purely on control input
        let velocity: Real = 2.0;
        for _ in 0..10 {
            fusion.predict(velocity).unwrap();
            // Corrupt inputs keep arriving from both sensors, but are completely ignored
            assert_eq!(
                fusion
                    .fuse(SensorMeasurement {
                        sensor_id: 1,
                        value: 999.0
                    })
                    .unwrap(),
                FusionStatus::IgnoredQuarantined
            );
            assert_eq!(
                fusion
                    .fuse(SensorMeasurement {
                        sensor_id: 2,
                        value: -999.0
                    })
                    .unwrap(),
                FusionStatus::IgnoredQuarantined
            );
        }

        // State smoothly tracked dead-reckoning trajectory (10 * 2.0 = 20.0) without crashing
        assert!((fusion.state() - 20.0).abs_val() < TOL);
        assert!(fusion.variance() > 1.0);
    }

    // =========================================================================
    // 5. C-FFI FDIR and Quarantine Safety
    // =========================================================================

    #[test]
    #[cfg(feature = "ffi")]
    fn test_c_ffi_fdir_quarantine_and_corrupted_covariance() {
        use aegis_filter::ffi::{
            aegis_fusion4_fuse, aegis_fusion4_get_sensor_health, aegis_fusion4_get_state,
            aegis_fusion4_init, aegis_fusion4_is_sensor_quarantined, aegis_fusion4_predict,
            aegis_fusion4_register_sensor, AegisFusion4, AegisFusionStatusCode, AegisStatusCode,
        };
        use core::mem::MaybeUninit;

        unsafe {
            let mut fusion = MaybeUninit::<AegisFusion4>::uninit();
            let status = aegis_fusion4_init(fusion.as_mut_ptr(), 10.0, 1.0, 0.01);
            assert_eq!(status, AegisStatusCode::Ok);
            let mut fusion = fusion.assume_init();

            // 1. Corrupted covariance input rejected via C-FFI
            let status = aegis_fusion4_register_sensor(&mut fusion, 1, -0.5, 2, 3.0, 0);
            assert_eq!(status, AegisStatusCode::InvalidParameter);

            // 2. Valid sensor registered
            let status = aegis_fusion4_register_sensor(&mut fusion, 1, 0.2, 2, 3.0, 2);
            assert_eq!(status, AegisStatusCode::Ok);
            assert_eq!(aegis_fusion4_is_sensor_quarantined(&fusion, 1), 0);

            // 3. Prediction
            assert_eq!(aegis_fusion4_predict(&mut fusion, 1.0), AegisStatusCode::Ok);

            // 4. Two extreme spikes -> Quarantined
            let mut f_status = AegisFusionStatusCode::UnregisteredSensor;
            assert_eq!(
                aegis_fusion4_fuse(&mut fusion, 1, 500.0, &mut f_status),
                AegisStatusCode::Ok
            );
            assert_eq!(f_status, AegisFusionStatusCode::Gated);

            assert_eq!(
                aegis_fusion4_fuse(&mut fusion, 1, 500.0, &mut f_status),
                AegisStatusCode::Ok
            );
            assert_eq!(f_status, AegisFusionStatusCode::Quarantined);
            assert_eq!(aegis_fusion4_is_sensor_quarantined(&fusion, 1), 1);
            assert_eq!(aegis_fusion4_get_sensor_health(&fusion, 1), 0.0);

            // 5. Subsequent spikes are IgnoredQuarantined
            assert_eq!(
                aegis_fusion4_fuse(&mut fusion, 1, 500.0, &mut f_status),
                AegisStatusCode::Ok
            );
            assert_eq!(f_status, AegisFusionStatusCode::IgnoredQuarantined);

            // State protected from 500m spike
            assert!((aegis_fusion4_get_state(&fusion) - 11.0).abs_val() < 0.5);
        }
    }
}
