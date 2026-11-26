//! aegis-filter: High-Integrity, Cyber-Resilient State Estimation & Robust Filtering Engine in pure Rust.
//!
//! # Quick Start Example
//! ```rust
//! use aegis_filter::ScalarKalmanFilter;
//!
//! let mut filter = ScalarKalmanFilter::new(0.0, 1.0, 0.01, 0.5).unwrap();
//! filter.predict(0.1).unwrap();
//! let state = filter.update(1.05, Some(3.0)).unwrap();
//! assert!((state - 0.7).abs() < 0.5);
//! ```
//!
//! # Architecture
//! - [`linear`]: Classical and Joseph-stabilized linear Kalman filters with Mahalanobis gating.
//! - [`unscented`]: Unscented Kalman Filter (UKF) for non-linear systems without analytical Jacobians.
//! - [`robust`]: $H_\infty$ (H-infinity) Minimax filters for bounded worst-case disturbances and online adaptive noise estimation.
//! - [`common`]: Core error definitions and `#![no_std]` deterministic mathematical traits.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

pub mod common;
pub mod eskf;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod fusion;
pub mod linear;
pub mod robust;
pub mod simd;
pub mod stats;
pub mod unscented;

pub use common::{
    FilterError, FloatOps, Real, DEFAULT_MAX_CONDITION_NUMBER, DEFAULT_OUTLIER_GATE_SIGMA,
};
pub use eskf::{
    ErrorStateKalmanFilter, EskfNoiseParameters, EskfNominalState, Quaternion, Vector3,
    EARTH_GRAVITY_Z_UP,
};
pub use fusion::{FusionStatus, MultiSensorFusion1D, SensorMeasurement, SensorSource};
pub use linear::{
    check_matrix_conditioning, invert_matrix, matrix_condition_number, matrix_norm_1,
    matrix_norm_frobenius, matrix_norm_inf, LinearKalmanFilter, ScalarKalmanFilter,
};
pub use robust::{AdaptiveFilter1D, HInfinityFilter1D};
pub use simd::{joseph_form_covariance_update, matmul, matmul_transpose_b, propagate_covariance};
pub use stats::WindowedStats;
pub use unscented::UnscentedKalmanFilter2D;
