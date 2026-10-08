//! Robust & Adaptive State Estimation.
//!
//! Features:
//! - **H-infinity (H_inf) Bounded-Error Filter**: Minimizes worst-case estimation error
//!   regardless of noise probability distribution, making it impervious to non-Gaussian
//!   sensor anomalies and adversarial attacks.
//! - **Adaptive Innovation Noise Estimator**: Automatically tunes measurement noise (R)
//!   at runtime using residual variance matching.

use crate::common::FilterError;

/// 1D Robust H-infinity (H_inf) Filter.
///
/// Unlike the Kalman Filter, which minimizes the *mean-square error* under the assumption
/// of known Gaussian noise, the H-infinity filter solves a minimax game:
/// it minimizes the *worst-case estimation error* over all possible bounded disturbances.
#[derive(Debug, Clone, Copy)]
pub struct HInfinityFilter1D {
    /// State estimate (x)
    x: f64,
    /// Covariance / error bound parameter (P)
    p: f64,
    /// Process disturbance bound (Q)
    q: f64,
    /// Measurement disturbance bound (R)
    r: f64,
    /// Performance attenuation bound (gamma). Smaller values enforce tighter worst-case error bounds.
    gamma: f64,
}

impl HInfinityFilter1D {
    /// Creates a new H-infinity filter.
    /// `gamma` must be strictly greater than 0 (typically >= 1.0).
    pub fn new(
        initial_state: f64,
        initial_variance: f64,
        process_bound_q: f64,
        measurement_bound_r: f64,
        gamma: f64,
    ) -> Result<Self, FilterError> {
        if !initial_state.is_finite()
            || initial_variance <= 0.0
            || process_bound_q < 0.0
            || measurement_bound_r <= 0.0
            || gamma <= 0.0
        {
            return Err(FilterError::InvalidCovariance);
        }

        Ok(Self {
            x: initial_state,
            p: initial_variance,
            q: process_bound_q,
            r: measurement_bound_r,
            gamma,
        })
    }

    /// Time update:
    /// x = x + u
    /// P = P + Q
    pub fn predict(&mut self, control_input: f64) -> Result<(), FilterError> {
        if !control_input.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }
        self.x += control_input;
        self.p += self.q;
        Ok(())
    }

    /// Measurement update under the H-infinity minimax criteria:
    ///
    /// Stability condition requires: `1 - gamma^(-2) * P + (H^2 * P) / R > 0`
    pub fn update(&mut self, measurement: f64) -> Result<f64, FilterError> {
        if !measurement.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }

        let gamma_sq = self.gamma * self.gamma;
        let _h_term = 1.0; // H = 1.0 for direct scalar measurement

        // H-infinity denominator term:
        // D = 1 - (1.0 / gamma_sq) * P + (H^2 * P) / R
        let d = 1.0 - (self.p / gamma_sq) + (self.p / self.r);

        // If the H-infinity bound is violated, the system cannot guarantee bounded error
        if d <= 1e-12 || !d.is_finite() {
            return Err(FilterError::NumericalInstability);
        }

        // H-infinity Gain:
        // K = P / (R * D)
        let k = self.p / (self.r * d);

        // State update
        let residual = measurement - self.x;
        self.x += k * residual;

        // Covariance update: P = P / D
        self.p /= d;

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
    pub fn bound(&self) -> f64 {
        self.p
    }
}

/// Adaptive Innovation Noise Estimator.
/// Automatically adjusts measurement variance R online using an exponential moving average
/// of the innovation sequence (Sage-Husa inspired).
#[derive(Debug, Clone)]
pub struct AdaptiveFilter1D {
    pub x: f64,
    pub p: f64,
    pub q: f64,
    pub r: f64,
    /// Smoothing factor for online noise estimation (0.01 to 0.1)
    pub alpha: f64,
}

impl AdaptiveFilter1D {
    pub fn new(initial_x: f64, initial_p: f64, q: f64, initial_r: f64, adaptation_rate: f64) -> Result<Self, FilterError> {
        if !initial_x.is_finite() || initial_p <= 0.0 || q < 0.0 || initial_r <= 0.0 || adaptation_rate <= 0.0 || adaptation_rate >= 1.0 {
            return Err(FilterError::InvalidCovariance);
        }

        Ok(Self {
            x: initial_x,
            p: initial_p,
            q,
            r: initial_r,
            alpha: adaptation_rate,
        })
    }

    pub fn predict(&mut self, u: f64) {
        self.x += u;
        self.p += self.q;
    }

    pub fn update(&mut self, z: f64) -> Result<f64, FilterError> {
        if !z.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }

        let y = z - self.x;
        let s = self.p + self.r;

        let k = self.p / s;
        self.x += k * y;

        // Joseph form update
        let one_minus_k = 1.0 - k;
        self.p = (one_minus_k * one_minus_k * self.p) + (k * k * self.r);

        // Adaptive estimation of R:
        // Instantaneous residual variance estimation: R_est = y^2 - P
        let r_sample = y * y - self.p;
        if r_sample > 1e-6 {
            self.r = (1.0 - self.alpha) * self.r + self.alpha * r_sample;
        }

        Ok(self.x)
    }

    #[inline(always)]
    pub fn estimated_measurement_noise(&self) -> f64 {
        self.r
    }
}
