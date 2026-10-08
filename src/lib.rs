//! High-Integrity, `#![no_std]` capable Kalman Filter Engine in pure Rust.
//!
//! # Design & Security Principles
//! - **Strict Validation**: Guard against `NaN`, infinite floats, non-positive variances, and ill-conditioned systems.
//! - **Numerical Stability**: Uses Joseph Form covariance update (`P = (I - KH)P(I - KH)^T + KRK^T`)
//!   to guarantee covariance symmetry and positive semi-definiteness even under finite precision floating point errors.
//! - **Zero Panics**: All operations return `Result<T, KalmanError>` instead of panicking.
//! - **Zero Allocation**: Pure `#![no_std]` core with const generics for fixed-size state & measurement vectors.
//! - **Mahalanobis Distance / Chi-Square Gating**: Outlier rejection for robust fault/anomaly detection in noisy environments.

#![cfg_attr(not(feature = "std"), no_std)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KalmanError {
    /// Provided float was NaN or infinite.
    NonFiniteInput,
    /// Provided variance or covariance matrix is non-positive or singular.
    InvalidCovariance,
    /// Innovation covariance is singular or numerically non-invertible.
    SingularInnovation,
    /// Measurement rejected by Chi-square / Mahalanobis gating test.
    MeasurementGated,
}

impl core::fmt::Display for KalmanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteInput => write!(f, "Input contains non-finite floating point values (NaN or Inf)"),
            Self::InvalidCovariance => write!(f, "Covariance is non-positive or poorly conditioned"),
            Self::SingularInnovation => write!(f, "Innovation covariance is singular or ill-conditioned for inversion"),
            Self::MeasurementGated => write!(f, "Measurement exceeded gating threshold and was rejected as an outlier"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for KalmanError {}

/// Trait verifying float sanity and providing no_std math utilities.
pub trait FloatOps {
    fn is_valid_float(&self) -> bool;
    fn abs_val(self) -> f64;
    fn sqrt_val(self) -> f64;
}

impl FloatOps for f64 {
    #[inline(always)]
    fn is_valid_float(&self) -> bool {
        self.is_finite()
    }

    #[inline(always)]
    fn abs_val(self) -> f64 {
        f64::from_bits(self.to_bits() & !(1u64 << 63))
    }

    /// High-precision square root utilizing standard hardware intrinsic when available,
    /// or high-convergence reciprocal square root / Newton-Raphson fallback in `#![no_std]`.
    #[inline]
    fn sqrt_val(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            if self <= 0.0 {
                return 0.0;
            }
            if !self.is_finite() {
                return self;
            }
            // High-convergence Newton-Raphson approximation
            let mut curr = self;
            for _ in 0..24 {
                let prev = curr;
                curr = 0.5 * (curr + self / curr);
                if (curr - prev).abs_val() <= 1e-15 * curr {
                    break;
                }
            }
            curr
        }
    }
}

/// 1D Scalar Kalman Filter designed for single-variable tracking
/// (e.g., temperature sensing, pressure altitude, simple sensor fusion).
#[derive(Debug, Clone, Copy)]
pub struct ScalarKalmanFilter {
    /// State estimate (x)
    x: f64,
    /// Estimate error covariance (P)
    p: f64,
    /// Process noise covariance (Q)
    q: f64,
    /// Measurement noise covariance (R)
    r: f64,
}

impl ScalarKalmanFilter {
    /// Constructs a new scalar Kalman Filter with validation.
    pub fn new(initial_state: f64, initial_variance: f64, process_noise: f64, measurement_noise: f64) -> Result<Self, KalmanError> {
        if !initial_state.is_finite()
            || !initial_variance.is_finite()
            || !process_noise.is_finite()
            || !measurement_noise.is_finite()
        {
            return Err(KalmanError::NonFiniteInput);
        }

        if initial_variance <= 0.0 || process_noise < 0.0 || measurement_noise <= 0.0 {
            return Err(KalmanError::InvalidCovariance);
        }

        Ok(Self {
            x: initial_state,
            p: initial_variance,
            q: process_noise,
            r: measurement_noise,
        })
    }

    /// Predicts the state forward one step.
    ///
    /// Form:
    /// x = x + u
    /// P = P + Q
    pub fn predict(&mut self, control_input: f64) -> Result<(), KalmanError> {
        if !control_input.is_finite() {
            return Err(KalmanError::NonFiniteInput);
        }

        self.x += control_input;
        self.p += self.q;

        if !self.x.is_finite() || !self.p.is_finite() {
            return Err(KalmanError::NonFiniteInput);
        }

        Ok(())
    }

    /// Updates the state using a measurement.
    /// Uses Joseph-stabilized variance update for scalar floats:
    /// `P = (1 - K)^2 * P + K^2 * R`
    pub fn update(&mut self, measurement: f64, gate_threshold_sigma: Option<f64>) -> Result<f64, KalmanError> {
        if !measurement.is_finite() {
            return Err(KalmanError::NonFiniteInput);
        }

        // Innovation / Residual: y = z - x
        let y = measurement - self.x;

        // Innovation variance: S = P + R
        let s = self.p + self.r;
        if s <= 1e-15 || !s.is_finite() {
            return Err(KalmanError::SingularInnovation);
        }

        // Optional outlier gating check
        if let Some(gate_sigma) = gate_threshold_sigma {
            if gate_sigma > 0.0 {
                let normalized_residual = y.abs_val() / s.sqrt_val();
                if normalized_residual > gate_sigma {
                    return Err(KalmanError::MeasurementGated);
                }
            }
        }

        // Optimal Kalman Gain: K = P / S
        let k = self.p / s;

        // Update state estimate: x = x + K * y
        self.x += k * y;

        // Joseph stabilized covariance update: P = (1 - K)^2 * P + K^2 * R
        // Guarantees positive variance even with floating point rounding artifacts
        let one_minus_k = 1.0 - k;
        self.p = (one_minus_k * one_minus_k * self.p) + (k * k * self.r);

        // Prevent zero or negative covariance from arithmetic underflow
        if self.p < 1e-15 {
            self.p = 1e-15;
        }

        Ok(self.x)
    }

    #[inline(always)]
    pub fn state(&self) -> f64 {
        self.x
    }

    #[inline(always)]
    pub fn variance(&self) -> f64 {
        self.p
    }

    #[inline(always)]
    pub fn set_process_noise(&mut self, q: f64) -> Result<(), KalmanError> {
        if !q.is_finite() || q < 0.0 {
            return Err(KalmanError::InvalidCovariance);
        }
        self.q = q;
        Ok(())
    }

    #[inline(always)]
    pub fn set_measurement_noise(&mut self, r: f64) -> Result<(), KalmanError> {
        if !r.is_finite() || r <= 0.0 {
            return Err(KalmanError::InvalidCovariance);
        }
        self.r = r;
        Ok(())
    }
}

/// Generic N-dimensional State, M-dimensional Measurement Linear Kalman Filter.
///
/// Designed with fixed-size const generics `STATES` and `MEASUREMENTS`
/// to operate without heap allocation (`#![no_std]`).
#[derive(Debug, Clone)]
pub struct KalmanFilter<const STATES: usize, const MEASUREMENTS: usize> {
    /// State estimate vector [STATES x 1]
    pub x: [f64; STATES],
    /// State covariance matrix [STATES x STATES]
    pub p: [[f64; STATES]; STATES],
    /// State transition matrix [STATES x STATES]
    pub f: [[f64; STATES]; STATES],
    /// Process noise covariance [STATES x STATES]
    pub q: [[f64; STATES]; STATES],
    /// Measurement matrix [MEASUREMENTS x STATES]
    pub h: [[f64; STATES]; MEASUREMENTS],
    /// Measurement noise covariance [MEASUREMENTS x MEASUREMENTS]
    pub r: [[f64; MEASUREMENTS]; MEASUREMENTS],
}

impl<const S: usize, const M: usize> KalmanFilter<S, M> {
    /// Creates a new Kalman filter instance with integrity verification.
    pub fn new(
        initial_x: [f64; S],
        initial_p: [[f64; S]; S],
        f: [[f64; S]; S],
        q: [[f64; S]; S],
        h: [[f64; S]; M],
        r: [[f64; M]; M],
    ) -> Result<Self, KalmanError> {
        // Sanity checks
        for val in initial_x.iter() {
            if !val.is_finite() {
                return Err(KalmanError::NonFiniteInput);
            }
        }
        Self::validate_matrix_finite(&initial_p)?;
        Self::validate_matrix_finite(&f)?;
        Self::validate_matrix_finite(&q)?;
        Self::validate_matrix_finite_rect(&h)?;
        Self::validate_matrix_finite(&r)?;

        // Validate diagonal positivity
        for i in 0..S {
            if initial_p[i][i] <= 0.0 || q[i][i] < 0.0 {
                return Err(KalmanError::InvalidCovariance);
            }
        }
        for j in 0..M {
            if r[j][j] <= 0.0 {
                return Err(KalmanError::InvalidCovariance);
            }
        }

        Ok(Self {
            x: initial_x,
            p: initial_p,
            f,
            q,
            h,
            r,
        })
    }

    #[inline]
    fn validate_matrix_finite<const ROWS: usize, const COLS: usize>(
        matrix: &[[f64; COLS]; ROWS],
    ) -> Result<(), KalmanError> {
        for row in matrix.iter() {
            for val in row.iter() {
                if !val.is_finite() {
                    return Err(KalmanError::NonFiniteInput);
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn validate_matrix_finite_rect<const ROWS: usize, const COLS: usize>(
        matrix: &[[f64; COLS]; ROWS],
    ) -> Result<(), KalmanError> {
        for row in matrix.iter() {
            for val in row.iter() {
                if !val.is_finite() {
                    return Err(KalmanError::NonFiniteInput);
                }
            }
        }
        Ok(())
    }

    /// Predict step:
    /// - x = F * x
    /// - P = F * P * F^T + Q
    pub fn predict(&mut self) -> Result<(), KalmanError> {
        // x_pred = F * x
        let mut x_pred = [0.0; S];
        for i in 0..S {
            let mut sum = 0.0;
            for j in 0..S {
                sum += self.f[i][j] * self.x[j];
            }
            x_pred[i] = sum;
        }

        // P_temp = F * P
        let mut f_p = [[0.0; S]; S];
        for i in 0..S {
            for j in 0..S {
                let mut sum = 0.0;
                for k in 0..S {
                    sum += self.f[i][k] * self.p[k][j];
                }
                f_p[i][j] = sum;
            }
        }

        // P_pred = f_p * F^T + Q
        let mut p_pred = [[0.0; S]; S];
        for i in 0..S {
            for j in 0..S {
                let mut sum = 0.0;
                for k in 0..S {
                    sum += f_p[i][k] * self.f[j][k]; // F^T[k][j] is F[j][k]
                }
                p_pred[i][j] = sum + self.q[i][j];
            }
        }

        // Ensure symmetry
        Self::symmetrize(&mut p_pred);

        self.x = x_pred;
        self.p = p_pred;

        Self::validate_matrix_finite(&self.p)?;
        for val in self.x.iter() {
            if !val.is_finite() {
                return Err(KalmanError::NonFiniteInput);
            }
        }

        Ok(())
    }

    /// Symmetrize square matrix: P = (P + P^T) / 2
    #[inline(always)]
    fn symmetrize<const N: usize>(matrix: &mut [[f64; N]; N]) {
        for i in 0..N {
            for j in (i + 1)..N {
                let avg = (matrix[i][j] + matrix[j][i]) * 0.5;
                matrix[i][j] = avg;
                matrix[j][i] = avg;
            }
        }
    }
}

/// Specialized 2D State / 1D Measurement Kalman Filter (Position + Velocity from 1D Position sensor).
/// Optimized with explicit closed-form scalar arithmetic for maximum performance and numerical safety.
#[derive(Debug, Clone)]
pub struct Kinematic1DFilter {
    /// State [position, velocity]
    pub x: [f64; 2],
    /// State covariance [2x2]
    pub p: [[f64; 2]; 2],
    /// Process noise intensity
    pub q_intensity: f64,
    /// Measurement noise variance (R)
    pub r: f64,
}

impl Kinematic1DFilter {
    pub fn new(
        initial_pos: f64,
        initial_vel: f64,
        pos_var: f64,
        vel_var: f64,
        q_intensity: f64,
        r_measurement_var: f64,
    ) -> Result<Self, KalmanError> {
        if !initial_pos.is_finite()
            || !initial_vel.is_finite()
            || !pos_var.is_finite()
            || !vel_var.is_finite()
            || !q_intensity.is_finite()
            || !r_measurement_var.is_finite()
        {
            return Err(KalmanError::NonFiniteInput);
        }

        if pos_var <= 0.0 || vel_var <= 0.0 || q_intensity < 0.0 || r_measurement_var <= 0.0 {
            return Err(KalmanError::InvalidCovariance);
        }

        Ok(Self {
            x: [initial_pos, initial_vel],
            p: [
                [pos_var, 0.0],
                [0.0, vel_var],
            ],
            q_intensity,
            r: r_measurement_var,
        })
    }

    /// Predicts state given elapsed delta time dt.
    pub fn predict(&mut self, dt: f64) -> Result<(), KalmanError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(KalmanError::NonFiniteInput);
        }

        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt2 * dt2;

        // F = [[1, dt], [0, 1]]
        // x = F * x
        self.x[0] += self.x[1] * dt;

        // Discrete continuous white noise model Q:
        let q11 = 0.25 * dt4 * self.q_intensity;
        let q12 = 0.5 * dt3 * self.q_intensity;
        let q22 = dt2 * self.q_intensity;

        // P_pred = F * P * F^T + Q
        let p00 = self.p[0][0];
        let p01 = self.p[0][1];
        let p10 = self.p[1][0];
        let p11 = self.p[1][1];

        let new_p00 = p00 + dt * (p10 + p01) + dt2 * p11 + q11;
        let new_p01 = p01 + dt * p11 + q12;
        let new_p10 = new_p01; // Symmetrized
        let new_p11 = p11 + q22;

        self.p[0][0] = new_p00;
        self.p[0][1] = new_p01;
        self.p[1][0] = new_p10;
        self.p[1][1] = new_p11;

        if !self.x[0].is_finite() || !self.x[1].is_finite() {
            return Err(KalmanError::NonFiniteInput);
        }

        Ok(())
    }

    /// Measurement update using Joseph stabilized form for 2-state / 1-measurement filter.
    pub fn update(&mut self, measured_pos: f64, gate_sigma: Option<f64>) -> Result<[f64; 2], KalmanError> {
        if !measured_pos.is_finite() {
            return Err(KalmanError::NonFiniteInput);
        }

        // H = [1.0, 0.0]
        // Innovation: y = z - H * x = z - x[0]
        let y = measured_pos - self.x[0];

        // S = H * P * H^T + R = P[0][0] + R
        let s = self.p[0][0] + self.r;
        if s <= 1e-15 || !s.is_finite() {
            return Err(KalmanError::SingularInnovation);
        }

        // Outlier rejection test
        if let Some(sigma) = gate_sigma {
            if sigma > 0.0 {
                let norm_residual = y.abs_val() / s.sqrt_val();
                if norm_residual > sigma {
                    return Err(KalmanError::MeasurementGated);
                }
            }
        }

        // Kalman Gain: K = P * H^T / S = [P00 / S, P10 / S]
        let k0 = self.p[0][0] / s;
        let k1 = self.p[1][0] / s;

        // State update: x = x + K * y
        self.x[0] += k0 * y;
        self.x[1] += k1 * y;

        // Joseph stabilized update:
        // P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        // Let A = I - K*H = [[1 - k0, 0], [-k1, 1]]
        let a00 = 1.0 - k0;
        let a01 = 0.0;
        let a10 = -k1;
        let a11 = 1.0;

        // AP = A * P
        let ap00 = a00 * self.p[0][0] + a01 * self.p[1][0];
        let ap01 = a00 * self.p[0][1] + a01 * self.p[1][1];
        let ap10 = a10 * self.p[0][0] + a11 * self.p[1][0];
        let ap11 = a10 * self.p[0][1] + a11 * self.p[1][1];

        // AP * A^T
        let apa00 = ap00 * a00 + ap01 * a01;
        let apa01 = ap00 * a10 + ap01 * a11;
        let apa10 = ap10 * a00 + ap11 * a01;
        let apa11 = ap10 * a10 + ap11 * a11;

        // K * R * K^T
        let krk00 = k0 * k0 * self.r;
        let krk01 = k0 * k1 * self.r;
        let krk10 = krk01;
        let krk11 = k1 * k1 * self.r;

        // Final P:
        let mut p_updated = [
            [apa00 + krk00, (apa01 + krk01 + apa10 + krk10) * 0.5],
            [(apa01 + krk01 + apa10 + krk10) * 0.5, apa11 + krk11],
        ];

        // Floor diagonal terms to prevent negative variance underflow
        if p_updated[0][0] < 1e-15 {
            p_updated[0][0] = 1e-15;
        }
        if p_updated[1][1] < 1e-15 {
            p_updated[1][1] = 1e-15;
        }

        self.p = p_updated;
        Ok(self.x)
    }

    #[inline(always)]
    pub fn position(&self) -> f64 {
        self.x[0]
    }

    #[inline(always)]
    pub fn velocity(&self) -> f64 {
        self.x[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_filter_convergence() {
        let mut kf = ScalarKalmanFilter::new(0.0, 10.0, 0.01, 1.0).unwrap();
        let ground_truth = 25.0;

        for _ in 0..100 {
            kf.predict(0.0).unwrap();
            let _ = kf.update(ground_truth + 0.1, None).unwrap();
        }

        assert!((kf.state() - ground_truth).abs() < 0.2);
        assert!(kf.variance() < 0.5);
    }

    #[test]
    fn test_scalar_filter_outlier_rejection() {
        let mut kf = ScalarKalmanFilter::new(10.0, 1.0, 0.01, 1.0).unwrap();

        // 3-sigma gating
        let outlier = 500.0;
        let res = kf.update(outlier, Some(3.0));
        assert_eq!(res, Err(KalmanError::MeasurementGated));
        // State remains unaffected
        assert_eq!(kf.state(), 10.0);
    }

    #[test]
    fn test_nan_inf_rejection() {
        assert_eq!(
            ScalarKalmanFilter::new(f64::NAN, 1.0, 0.1, 1.0).err(),
            Some(KalmanError::NonFiniteInput)
        );
        let mut kf = ScalarKalmanFilter::new(0.0, 1.0, 0.1, 1.0).unwrap();
        assert_eq!(
            kf.update(f64::INFINITY, None).err(),
            Some(KalmanError::NonFiniteInput)
        );
    }

    #[test]
    fn test_kinematic_1d_tracking() {
        // True motion: starts at 0, moving at 2.0 m/s
        let mut kf = Kinematic1DFilter::new(0.0, 0.0, 1.0, 1.0, 0.05, 0.5).unwrap();

        let dt = 0.1;
        let mut true_pos = 0.0;
        let true_vel = 2.0;

        for _ in 0..50 {
            true_pos += true_vel * dt;
            kf.predict(dt).unwrap();
            let noisy_meas = true_pos + 0.05;
            let _ = kf.update(noisy_meas, None).unwrap();
        }

        assert!((kf.position() - true_pos).abs() < 0.3);
        assert!((kf.velocity() - true_vel).abs() < 0.3);
    }
}
