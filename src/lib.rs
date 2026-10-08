//! aegis-filter: High-Integrity, Cyber-Resilient State Estimation & Robust Filtering Engine in pure Rust.
//!
//! # Architecture
//! - [`linear`]: Classical and Joseph-stabilized linear Kalman filters with Mahalanobis gating.
//! - [`unscented`]: Unscented Kalman Filter (UKF) for non-linear systems without analytical Jacobians.
//! - [`robust`]: $H_\infty$ (H-infinity) Minimax filters for bounded worst-case disturbances and online adaptive noise estimation.
//! - [`common`]: Core error definitions and `#![no_std]` deterministic mathematical traits.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod common;
pub mod linear;
pub mod unscented;
pub mod robust;

pub use common::FilterError;
pub use linear::{ScalarKalmanFilter, LinearKalmanFilter};
pub use unscented::UnscentedKalmanFilter2D;
pub use robust::{HInfinityFilter1D, AdaptiveFilter1D};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unscented_filter_tracking() {
        // Non-linear tracking: range sensor observing state
        let mut ukf = UnscentedKalmanFilter2D::new(
            [1.0, 0.5],
            [[0.5, 0.0], [0.0, 0.5]],
            [[0.01, 0.0], [0.0, 0.01]],
            0.1,
        )
        .unwrap();

        // Predict with simple velocity integration
        ukf.predict(0.1, |x, dt| [x[0] + x[1] * dt, x[1]]).unwrap();

        // Update with non-linear quadratic observation h(x) = x0^2
        let measurement = 1.1 * 1.1;
        let res = ukf.update(measurement, |x| x[0] * x[0], None).unwrap();

        assert!((res[0] - 1.1).abs() < 0.2);
    }

    #[test]
    fn test_h_infinity_robustness() {
        // H-infinity bounded filter under adversarial shock
        let mut h_inf = HInfinityFilter1D::new(10.0, 2.0, 0.05, 0.5, 5.0).unwrap();

        for _ in 0..10 {
            h_inf.predict(0.0).unwrap();
            let _ = h_inf.update(10.2).unwrap();
        }

        assert!((h_inf.state() - 10.2).abs() < 0.3);
        assert!(h_inf.bound() > 0.0);
    }
}
