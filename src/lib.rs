//! aegis-filter: High-Integrity, Cyber-Resilient State Estimation & Robust Filtering Engine in pure Rust.
//!
//! # Architecture
//! - [`linear`]: Classical and Joseph-stabilized linear Kalman filters with Mahalanobis gating.
//! - [`unscented`]: Unscented Kalman Filter (UKF) for non-linear systems without analytical Jacobians.
//! - [`robust`]: $H_\infty$ (H-infinity) Minimax filters for bounded worst-case disturbances and online adaptive noise estimation.
//! - [`common`]: Core error definitions and `#![no_std]` deterministic mathematical traits.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod common;
pub mod fusion;
pub mod linear;
pub mod robust;
pub mod stats;
pub mod unscented;

pub use common::FilterError;
pub use fusion::{FusionStatus, MultiSensorFusion1D, SensorMeasurement, SensorSource};
pub use linear::{LinearKalmanFilter, ScalarKalmanFilter};
pub use robust::{AdaptiveFilter1D, HInfinityFilter1D};
pub use stats::WindowedStats;
pub use unscented::UnscentedKalmanFilter2D;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_filter_convergence() {
        let mut filter = ScalarKalmanFilter::new(0.0, 10.0, 0.01, 0.5).unwrap();
        let target = 25.0;
        for _ in 0..50 {
            filter.predict(0.0).unwrap();
            let _ = filter.update(target + 0.05, None).unwrap();
        }
        assert!((filter.state() - target).abs() < 0.2);
    }

    #[test]
    fn test_scalar_filter_outlier_gating() {
        let mut filter = ScalarKalmanFilter::new(10.0, 1.0, 0.01, 0.5).unwrap();
        // Adversarial spike — must be rejected by 3-sigma gating
        let err = filter.update(9999.0, Some(3.0));
        assert_eq!(err, Err(FilterError::MeasurementGated));
        assert!((filter.state() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ukf_linear_observation_multi_step() {
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
    fn test_ukf_nonlinear_tight_covariance() {
        // h(x) = x[0]^2 — non-linear. With tight initial covariance sigma
        // points stay close to mean, so linearisation error is small and a
        // single update step visibly moves the state toward the true value.
        let true_pos: f64 = 3.0;
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
    fn test_h_infinity_robustness() {
        let mut h_inf = HInfinityFilter1D::new(10.0, 1.0, 0.05, 0.5, 5.0).unwrap();

        for _ in 0..20 {
            h_inf.predict(0.0).unwrap();
            let _ = h_inf.update(10.1).unwrap();
        }

        assert!((h_inf.state() - 10.1).abs() < 0.3);
        assert!(h_inf.bound() > 0.0);
    }

    #[test]
    fn test_adaptive_filter_noise_tracking() {
        let mut af = AdaptiveFilter1D::new(0.0, 5.0, 0.01, 1.0, 0.05).unwrap();
        for _ in 0..40 {
            af.predict(0.0);
            let _ = af.update(20.0).unwrap();
        }
        assert!((af.x - 20.0).abs() < 0.5, "af.x = {}", af.x);
    }

    #[test]
    fn test_nan_inf_rejection() {
        let result = ScalarKalmanFilter::new(f64::NAN, 1.0, 0.1, 1.0);
        assert_eq!(result.err(), Some(FilterError::NonFiniteInput));

        let mut f = ScalarKalmanFilter::new(0.0, 1.0, 0.1, 1.0).unwrap();
        assert_eq!(
            f.update(f64::INFINITY, None).err(),
            Some(FilterError::NonFiniteInput)
        );
    }

    #[test]
    fn test_multi_sensor_fusion_and_quarantine() {
        // Setup fusion engine with capacity for 3 sensors
        let mut fusion = MultiSensorFusion1D::<3>::new(0.0, 5.0, 0.05).unwrap();

        // Sensor 1: Accurate Barometer (R=0.2, max 3 anomalies)
        let baro = SensorSource::new(1, 0.2, 3).unwrap();
        // Sensor 2: Noisy GPS (R=1.5, max 2 anomalies)
        let gps = SensorSource::new(2, 1.5, 2).unwrap();

        fusion.register_sensor(baro).unwrap();
        fusion.register_sensor(gps).unwrap();

        // Nominal fusion
        fusion.predict(1.0).unwrap();
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 1.05,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Fused);

        // Adversarial spoofing on GPS (id=2)
        // Attack 1: Gated
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 2,
                value: 500.0,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Gated);

        // Attack 2: Second consecutive anomaly triggers Quarantine!
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 2,
                value: 500.0,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Quarantined);
        assert_eq!(fusion.sensor_health(2), Some(0.0));

        // Subsequent measurements from quarantined GPS are safely ignored
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 2,
                value: 1.0,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::IgnoredQuarantined);

        // Barometer continues nominal operation unhindered
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 1.02,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Fused);
    }

    #[test]
    fn test_sensor_auto_recovery() {
        let mut fusion = MultiSensorFusion1D::<2>::new(10.0, 1.0, 0.01).unwrap();
        // Sensor with auto-recovery threshold of 2 consecutive healthy observations
        let gps = SensorSource::new(1, 0.5, 2).unwrap().with_auto_recovery(2);
        fusion.register_sensor(gps).unwrap();

        // 2 anomalies force quarantine
        let _ = fusion.fuse(SensorMeasurement {
            sensor_id: 1,
            value: 100.0,
        });
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 100.0,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Quarantined);

        // 1st healthy sample while quarantined
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 10.05,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::IgnoredQuarantined);

        // 2nd healthy sample triggers auto-recovery!
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 10.02,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Recovered);

        // Subsequent sample is actively fused again
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 10.01,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Fused);
    }

    #[test]
    fn test_windowed_stats_rolling_metrics() {
        let mut stats = WindowedStats::<4>::new();
        assert!(stats.is_empty());
        assert_eq!(stats.len(), 0);

        stats.push(2.0).unwrap();
        stats.push(4.0).unwrap();
        assert_eq!(stats.len(), 2);
        assert!((stats.mean().unwrap() - 3.0).abs() < 1e-9);
        assert_eq!(stats.min(), Some(2.0));
        assert_eq!(stats.max(), Some(4.0));
        assert!((stats.peak_to_peak().unwrap() - 2.0).abs() < 1e-9);

        // Push more samples to exceed window capacity of 4
        stats.push(6.0).unwrap();
        stats.push(8.0).unwrap();
        assert_eq!(stats.len(), 4);
        assert!((stats.mean().unwrap() - 5.0).abs() < 1e-9);

        // Overwrite oldest (2.0) with 10.0 -> window has [10, 4, 6, 8], mean = 7.0
        stats.push(10.0).unwrap();
        assert_eq!(stats.len(), 4);
        assert!((stats.mean().unwrap() - 7.0).abs() < 1e-9);
        assert_eq!(stats.min(), Some(4.0));
        assert_eq!(stats.max(), Some(10.0));
        assert!(stats.variance().unwrap() > 0.0);
    }

    #[test]
    fn test_fdir_consecutive_anomaly_reset() {
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
        assert_eq!(fusion.sensor_health(1), Some(0.55));
    }

    #[test]
    fn test_fusion_singularity_guard() {
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
