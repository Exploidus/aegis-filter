//! C-compatible Foreign Function Interface (FFI) for NASA F Prime and C/C++ Flight Software.
//!
//! Exposes zero-allocation, deterministic `#![no_std]` state estimation and FDIR pipelines
//! with C-compatible ABI (`extern "C"`).

#![allow(clippy::missing_safety_doc)]

use crate::common::FilterError;
use crate::fusion::{FusionStatus, MultiSensorFusion1D, SensorMeasurement, SensorSource};
use crate::linear::ScalarKalmanFilter;
use crate::robust::HInfinityFilter1D;

/// Status return codes for C API calls.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AegisStatusCode {
    /// Operation completed successfully.
    Ok = 0,
    /// Provided float was NaN or infinite.
    NonFiniteInput = -1,
    /// Provided covariance is non-positive or ill-conditioned.
    InvalidCovariance = -2,
    /// Innovation covariance is singular or ill-conditioned.
    SingularInnovation = -3,
    /// Measurement was rejected by Mahalanobis / chi-square gating.
    MeasurementGated = -4,
    /// Riccati denominator bound was violated ($H_\infty$ minimax infeasibility).
    NumericalInstability = -5,
    /// Invalid parameter value provided.
    InvalidParameter = -6,
    /// Maximum sensor array capacity exceeded.
    CapacityExceeded = -7,
    /// Null pointer was passed to the C API.
    NullPointer = -8,
}

impl From<FilterError> for AegisStatusCode {
    fn from(err: FilterError) -> Self {
        match err {
            FilterError::NonFiniteInput => Self::NonFiniteInput,
            FilterError::InvalidCovariance => Self::InvalidCovariance,
            FilterError::SingularInnovation => Self::SingularInnovation,
            FilterError::MeasurementGated => Self::MeasurementGated,
            FilterError::NumericalInstability => Self::NumericalInstability,
            FilterError::InvalidParameter => Self::InvalidParameter,
            FilterError::CapacityExceeded => Self::CapacityExceeded,
        }
    }
}

/// Status of multi-sensor fusion measurement update.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AegisFusionStatusCode {
    /// Measurement accepted and fused into state.
    Fused = 0,
    /// Measurement rejected by innovation gating threshold.
    Gated = 1,
    /// Sensor quarantined due to consecutive anomalies exceeding threshold.
    Quarantined = 2,
    /// Measurement ignored because sensor is actively quarantined.
    IgnoredQuarantined = 3,
    /// Quarantined sensor observed consecutive healthy samples and recovered.
    Recovered = 4,
    /// Sensor ID was not registered in the filter.
    UnregisteredSensor = 5,
}

impl From<FusionStatus> for AegisFusionStatusCode {
    fn from(status: FusionStatus) -> Self {
        match status {
            FusionStatus::Fused => Self::Fused,
            FusionStatus::Gated => Self::Gated,
            FusionStatus::Quarantined => Self::Quarantined,
            FusionStatus::IgnoredQuarantined => Self::IgnoredQuarantined,
            FusionStatus::Recovered => Self::Recovered,
            FusionStatus::UnregisteredSensor => Self::UnregisteredSensor,
        }
    }
}

// ----------------------------------------------------------------------------
// 1. Scalar Kalman Filter C-FFI
// ----------------------------------------------------------------------------

/// C-compatible handle for 1D Scalar Kalman Filter.
#[repr(C)]
pub struct AegisScalarFilter {
    inner: ScalarKalmanFilter,
}

/// Initializes a new Scalar Kalman Filter instance into the memory pointed to by `filter`.
#[no_mangle]
pub unsafe extern "C" fn aegis_scalar_filter_init(
    filter: *mut AegisScalarFilter,
    initial_state: f64,
    initial_variance: f64,
    process_noise_q: f64,
    measurement_noise_r: f64,
) -> AegisStatusCode {
    if filter.is_null() {
        return AegisStatusCode::NullPointer;
    }

    match ScalarKalmanFilter::new(
        initial_state,
        initial_variance,
        process_noise_q,
        measurement_noise_r,
    ) {
        Ok(kf) => {
            (*filter).inner = kf;
            AegisStatusCode::Ok
        }
        Err(e) => e.into(),
    }
}

/// Propagates state and covariance forward using physical control input.
#[no_mangle]
pub unsafe extern "C" fn aegis_scalar_filter_predict(
    filter: *mut AegisScalarFilter,
    control_input: f64,
) -> AegisStatusCode {
    if filter.is_null() {
        return AegisStatusCode::NullPointer;
    }

    match (*filter).inner.predict(control_input) {
        Ok(()) => AegisStatusCode::Ok,
        Err(e) => e.into(),
    }
}

/// Default recommended 3-sigma innovation gating threshold for adversarial outlier rejection.
pub const AEGIS_DEFAULT_GATE_SIGMA: f64 = 3.0;

/// Sentinel value to explicitly disable innovation gating in C API calls.
pub const AEGIS_GATE_DISABLE: f64 = -1.0;

/// Configures persistent default outlier gating threshold on the scalar filter (fail-closed posture).
///
/// Pass a positive sigma threshold (e.g. `AEGIS_DEFAULT_GATE_SIGMA = 3.0`) to enforce gating.
/// Pass `AEGIS_GATE_DISABLE` (-1.0) or `0.0` to disable default gating.
#[no_mangle]
pub unsafe extern "C" fn aegis_scalar_filter_set_default_gate(
    filter: *mut AegisScalarFilter,
    gate_threshold_sigma: f64,
) -> AegisStatusCode {
    if filter.is_null() {
        return AegisStatusCode::NullPointer;
    }
    if gate_threshold_sigma > 0.0 {
        match (*filter).inner.with_default_gate(gate_threshold_sigma) {
            Ok(f) => {
                (*filter).inner = f;
                AegisStatusCode::Ok
            }
            Err(e) => e.into(),
        }
    } else {
        (*filter).inner = (*filter).inner.disable_default_gate();
        AegisStatusCode::Ok
    }
}

/// Fuses incoming measurement with optional Mahalanobis 3-sigma innovation gating.
///
/// # Security & Gating Semantics
/// - Passing a positive `gate_threshold_sigma` (e.g. `AEGIS_DEFAULT_GATE_SIGMA = 3.0`) enforces
///   outlier gating for this call. Outliers are rejected with [`AegisStatusCode::MeasurementGated`] (-4).
/// - Passing `AEGIS_GATE_DISABLE` (< 0.0) explicitly bypasses innovation gating for this update.
/// - Passing `0.0` falls back to the filter's configured default gate ([`aegis_scalar_filter_set_default_gate`]).
#[no_mangle]
pub unsafe extern "C" fn aegis_scalar_filter_update(
    filter: *mut AegisScalarFilter,
    measurement: f64,
    gate_threshold_sigma: f64,
    out_state: *mut f64,
) -> AegisStatusCode {
    if filter.is_null() {
        return AegisStatusCode::NullPointer;
    }

    let gate_opt = if gate_threshold_sigma > 0.0 {
        Some(gate_threshold_sigma)
    } else if gate_threshold_sigma < 0.0 {
        Some(-1.0)
    } else {
        None
    };

    match (*filter).inner.update(measurement, gate_opt) {
        Ok(state) => {
            if !out_state.is_null() {
                *out_state = state;
            }
            AegisStatusCode::Ok
        }
        Err(e) => e.into(),
    }
}

/// Returns current estimated state mean.
#[no_mangle]
pub unsafe extern "C" fn aegis_scalar_filter_get_state(filter: *const AegisScalarFilter) -> f64 {
    if filter.is_null() {
        return f64::NAN;
    }
    (*filter).inner.state()
}

/// Returns current estimated error variance.
#[no_mangle]
pub unsafe extern "C" fn aegis_scalar_filter_get_variance(filter: *const AegisScalarFilter) -> f64 {
    if filter.is_null() {
        return f64::NAN;
    }
    (*filter).inner.variance()
}

// ----------------------------------------------------------------------------
// 2. H-Infinity Minimax Robust Filter C-FFI
// ----------------------------------------------------------------------------

/// C-compatible handle for 1D Robust H-Infinity Filter.
#[repr(C)]
pub struct AegisHInfFilter {
    inner: HInfinityFilter1D,
}

/// Initializes an H-Infinity Filter instance into memory pointed to by `filter`.
#[no_mangle]
pub unsafe extern "C" fn aegis_h_inf_filter_init(
    filter: *mut AegisHInfFilter,
    initial_state: f64,
    initial_variance: f64,
    process_bound_q: f64,
    measurement_bound_r: f64,
    gamma: f64,
) -> AegisStatusCode {
    if filter.is_null() {
        return AegisStatusCode::NullPointer;
    }

    match HInfinityFilter1D::new(
        initial_state,
        initial_variance,
        process_bound_q,
        measurement_bound_r,
        gamma,
    ) {
        Ok(h_inf) => {
            (*filter).inner = h_inf;
            AegisStatusCode::Ok
        }
        Err(e) => e.into(),
    }
}

/// Propagates state forward under bounded disturbance.
#[no_mangle]
pub unsafe extern "C" fn aegis_h_inf_filter_predict(
    filter: *mut AegisHInfFilter,
    control_input: f64,
) -> AegisStatusCode {
    if filter.is_null() {
        return AegisStatusCode::NullPointer;
    }

    match (*filter).inner.predict(control_input) {
        Ok(()) => AegisStatusCode::Ok,
        Err(e) => e.into(),
    }
}

/// Updates state under discrete-time bounded-real Riccati criteria.
#[no_mangle]
pub unsafe extern "C" fn aegis_h_inf_filter_update(
    filter: *mut AegisHInfFilter,
    measurement: f64,
    out_state: *mut f64,
) -> AegisStatusCode {
    if filter.is_null() {
        return AegisStatusCode::NullPointer;
    }

    match (*filter).inner.update(measurement) {
        Ok(state) => {
            if !out_state.is_null() {
                *out_state = state;
            }
            AegisStatusCode::Ok
        }
        Err(e) => e.into(),
    }
}

/// Returns current H-Infinity state estimate.
#[no_mangle]
pub unsafe extern "C" fn aegis_h_inf_filter_get_state(filter: *const AegisHInfFilter) -> f64 {
    if filter.is_null() {
        return f64::NAN;
    }
    (*filter).inner.state()
}

/// Returns current H-Infinity error bound parameter (P).
#[no_mangle]
pub unsafe extern "C" fn aegis_h_inf_filter_get_bound(filter: *const AegisHInfFilter) -> f64 {
    if filter.is_null() {
        return f64::NAN;
    }
    (*filter).inner.bound()
}

// ----------------------------------------------------------------------------
// 3. Multi-Sensor FDIR Fusion Engine (4-Sensor Flight Array) C-FFI
// ----------------------------------------------------------------------------

/// C-compatible handle for 4-Sensor FDIR Fusion Engine.
#[repr(C)]
pub struct AegisFusion4 {
    inner: MultiSensorFusion1D<4>,
}

/// Initializes a 4-channel Multi-Sensor Fusion filter.
#[no_mangle]
pub unsafe extern "C" fn aegis_fusion4_init(
    fusion: *mut AegisFusion4,
    x0: f64,
    p0: f64,
    q: f64,
) -> AegisStatusCode {
    if fusion.is_null() {
        return AegisStatusCode::NullPointer;
    }

    match MultiSensorFusion1D::<4>::new(x0, p0, q) {
        Ok(engine) => {
            (*fusion).inner = engine;
            AegisStatusCode::Ok
        }
        Err(e) => e.into(),
    }
}

/// Registers a redundant sensor source into the fusion array.
#[no_mangle]
pub unsafe extern "C" fn aegis_fusion4_register_sensor(
    fusion: *mut AegisFusion4,
    sensor_id: u8,
    variance: f64,
    max_consecutive_anomalies: u32,
    gate_threshold_sigma: f64,
    auto_recovery_threshold: u32,
) -> AegisStatusCode {
    if fusion.is_null() {
        return AegisStatusCode::NullPointer;
    }

    let mut source = match SensorSource::new(sensor_id, variance, max_consecutive_anomalies) {
        Ok(s) => s,
        Err(e) => return e.into(),
    };

    if gate_threshold_sigma > 0.0 {
        match source.with_gate_threshold(gate_threshold_sigma) {
            Ok(s) => source = s,
            Err(e) => return e.into(),
        }
    }

    if auto_recovery_threshold > 0 {
        source = source.with_auto_recovery(auto_recovery_threshold);
    }

    match (*fusion).inner.register_sensor(source) {
        Ok(()) => AegisStatusCode::Ok,
        Err(e) => e.into(),
    }
}

/// Time update for multi-sensor state propagation.
#[no_mangle]
pub unsafe extern "C" fn aegis_fusion4_predict(
    fusion: *mut AegisFusion4,
    control_input: f64,
) -> AegisStatusCode {
    if fusion.is_null() {
        return AegisStatusCode::NullPointer;
    }

    match (*fusion).inner.predict(control_input) {
        Ok(()) => AegisStatusCode::Ok,
        Err(e) => e.into(),
    }
}

/// Fuses incoming measurement from a sensor source with automated FDIR checks.
#[no_mangle]
pub unsafe extern "C" fn aegis_fusion4_fuse(
    fusion: *mut AegisFusion4,
    sensor_id: u8,
    value: f64,
    out_status: *mut AegisFusionStatusCode,
) -> AegisStatusCode {
    if fusion.is_null() {
        return AegisStatusCode::NullPointer;
    }

    let meas = SensorMeasurement { sensor_id, value };
    match (*fusion).inner.fuse(meas) {
        Ok(status) => {
            if !out_status.is_null() {
                *out_status = status.into();
            }
            AegisStatusCode::Ok
        }
        Err(e) => e.into(),
    }
}

/// Returns fused state estimate.
#[no_mangle]
pub unsafe extern "C" fn aegis_fusion4_get_state(fusion: *const AegisFusion4) -> f64 {
    if fusion.is_null() {
        return f64::NAN;
    }
    (*fusion).inner.x
}

/// Returns fused error variance.
#[no_mangle]
pub unsafe extern "C" fn aegis_fusion4_get_variance(fusion: *const AegisFusion4) -> f64 {
    if fusion.is_null() {
        return f64::NAN;
    }
    (*fusion).inner.p
}

/// Returns real-time health score of a sensor (1.0 = nominal, 0.0 = quarantined).
#[no_mangle]
pub unsafe extern "C" fn aegis_fusion4_get_sensor_health(
    fusion: *const AegisFusion4,
    sensor_id: u8,
) -> f64 {
    if fusion.is_null() {
        return f64::NAN;
    }
    (*fusion).inner.sensor_health(sensor_id).unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::MaybeUninit;

    #[test]
    fn test_c_ffi_scalar_filter() {
        unsafe {
            // Null pointer safety
            assert_eq!(
                aegis_scalar_filter_init(core::ptr::null_mut(), 0.0, 1.0, 0.01, 0.5),
                AegisStatusCode::NullPointer
            );

            let mut filter = MaybeUninit::<AegisScalarFilter>::uninit();
            let status = aegis_scalar_filter_init(filter.as_mut_ptr(), 0.0, 1.0, 0.01, 0.5);
            assert_eq!(status, AegisStatusCode::Ok);
            let mut filter = filter.assume_init();

            let status = aegis_scalar_filter_predict(&mut filter, 0.1);
            assert_eq!(status, AegisStatusCode::Ok);

            let mut out = 0.0;
            let status = aegis_scalar_filter_update(&mut filter, 1.05, 3.0, &mut out);
            assert_eq!(status, AegisStatusCode::Ok);
            assert!(out > 0.0);
            assert!((aegis_scalar_filter_get_state(&filter) - out).abs() < 1e-12);
            assert!(aegis_scalar_filter_get_variance(&filter) > 0.0);

            // Rejection of adversary spike
            let status = aegis_scalar_filter_update(&mut filter, 9999.0, 3.0, &mut out);
            assert_eq!(status, AegisStatusCode::MeasurementGated);

            // Test fail-closed default gate configuration
            let status = aegis_scalar_filter_set_default_gate(&mut filter, 3.0);
            assert_eq!(status, AegisStatusCode::Ok);
            // Passing 0.0 falls back to configured 3.0 default gate -> rejects spike
            let status = aegis_scalar_filter_update(&mut filter, 9999.0, 0.0, &mut out);
            assert_eq!(status, AegisStatusCode::MeasurementGated);
            // Passing explicit AEGIS_GATE_DISABLE (-1.0) allows through
            let status = aegis_scalar_filter_update(&mut filter, 1.05, AEGIS_GATE_DISABLE, &mut out);
            assert_eq!(status, AegisStatusCode::Ok);
        }
    }

    #[test]
    fn test_c_ffi_h_infinity_filter() {
        unsafe {
            let mut hinf = MaybeUninit::<AegisHInfFilter>::uninit();
            let status = aegis_h_inf_filter_init(hinf.as_mut_ptr(), 0.0, 1.0, 0.01, 0.5, 5.0);
            assert_eq!(status, AegisStatusCode::Ok);
            let mut hinf = hinf.assume_init();

            let status = aegis_h_inf_filter_predict(&mut hinf, 0.1);
            assert_eq!(status, AegisStatusCode::Ok);

            let mut out = 0.0;
            let status = aegis_h_inf_filter_update(&mut hinf, 1.05, &mut out);
            assert_eq!(status, AegisStatusCode::Ok);
            assert!(out > 0.0);
            assert!(aegis_h_inf_filter_get_bound(&hinf) > 0.0);
        }
    }

    #[test]
    fn test_c_ffi_fusion4_and_fdir() {
        unsafe {
            let mut fusion = MaybeUninit::<AegisFusion4>::uninit();
            let status = aegis_fusion4_init(fusion.as_mut_ptr(), 0.0, 1.0, 0.01);
            assert_eq!(status, AegisStatusCode::Ok);
            let mut fusion = fusion.assume_init();

            // Register Sensor 1
            let status = aegis_fusion4_register_sensor(&mut fusion, 1, 0.2, 2, 3.0, 2);
            assert_eq!(status, AegisStatusCode::Ok);

            let status = aegis_fusion4_predict(&mut fusion, 0.5);
            assert_eq!(status, AegisStatusCode::Ok);

            let mut fusion_status = AegisFusionStatusCode::UnregisteredSensor;
            let status = aegis_fusion4_fuse(&mut fusion, 1, 0.52, &mut fusion_status);
            assert_eq!(status, AegisStatusCode::Ok);
            assert_eq!(fusion_status, AegisFusionStatusCode::Fused);
            assert!((aegis_fusion4_get_state(&fusion) - 0.52).abs() < 0.2);
            assert_eq!(aegis_fusion4_get_sensor_health(&fusion, 1), 1.0);
        }
    }
}
