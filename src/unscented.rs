//! Unscented Kalman Filter (UKF) for non-linear state estimation without calculating Jacobians.
//!
//! Evaluates non-linear transformations directly on a set of deterministically
//! chosen Sigma Points around the mean state.
//!
//! Formulated for `#![no_std]` execution without heap allocations using compile-time const generics.

use crate::common::{FilterError, FloatOps};

/// 2-State / 1-Measurement Unscented Kalman Filter.
///
/// Designed to replace linear KF and Extended Kalman Filter (EKF)
/// in non-linear tracking problems (e.g. radar range, pendulums, angle transformations).
#[derive(Debug, Clone)]
pub struct UnscentedKalmanFilter2D {
    /// State mean [x0, x1]
    pub x: [f64; 2],
    /// State covariance [2x2]
    pub p: [[f64; 2]; 2],
    /// Process noise covariance [2x2]
    pub q: [[f64; 2]; 2],
    /// Measurement noise variance (R)
    pub r: f64,

    // Merwe scaled unscented transform parameters:
    alpha: f64,
    beta: f64,
    kappa: f64,
}

impl UnscentedKalmanFilter2D {
    /// Initializes a new 2D Unscented Kalman Filter.
    pub fn new(
        initial_x: [f64; 2],
        initial_p: [[f64; 2]; 2],
        q: [[f64; 2]; 2],
        r: f64,
    ) -> Result<Self, FilterError> {
        if !initial_x[0].is_finite()
            || !initial_x[1].is_finite()
            || initial_p[0][0] <= 0.0
            || initial_p[1][1] <= 0.0
            || r <= 0.0
        {
            return Err(FilterError::InvalidCovariance);
        }

        Ok(Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
            alpha: 1e-3,
            beta: 2.0, // Optimal for Gaussian distributions
            kappa: 0.0,
        })
    }

    /// Computes 2*N + 1 = 5 Sigma Points and their corresponding mean and covariance weights.
    fn generate_sigma_points(&self) -> Result<([f64; 5], [f64; 5], [[f64; 2]; 5]), FilterError> {
        let n = 2.0;
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let c = n + lambda;
        if c <= 0.0 {
            return Err(FilterError::NumericalInstability);
        }

        // Cholesky square root of (c * P): L * L^T = c * P
        // For 2x2 matrix:
        let a = c * self.p[0][0];
        let b = c * self.p[0][1];
        let d = c * self.p[1][1];

        if a <= 1e-15 {
            return Err(FilterError::NumericalInstability);
        }
        let l00 = a.sqrt_val();
        let l10 = b / l00;
        let l11_sq = d - (l10 * l10);
        if l11_sq <= 1e-15 {
            return Err(FilterError::NumericalInstability);
        }
        let l11 = l11_sq.sqrt_val();

        // L columns: col0 = [l00, l10], col1 = [0, l11]
        let mut sigma_points = [[0.0; 2]; 5];
        sigma_points[0] = self.x;
        sigma_points[1] = [self.x[0] + l00, self.x[1] + l10];
        sigma_points[2] = [self.x[0], self.x[1] + l11];
        sigma_points[3] = [self.x[0] - l00, self.x[1] - l10];
        sigma_points[4] = [self.x[0], self.x[1] - l11];

        // Compute weights
        let mut wm = [0.0; 5];
        let mut wc = [0.0; 5];

        wm[0] = lambda / (n + lambda);
        wc[0] = wm[0] + (1.0 - self.alpha * self.alpha + self.beta);

        let weight_i = 1.0 / (2.0 * (n + lambda));
        for i in 1..5 {
            wm[i] = weight_i;
            wc[i] = weight_i;
        }

        Ok((wm, wc, sigma_points))
    }

    /// Predicts the state forward through an arbitrary non-linear dynamic transition function `f(x, dt)`.
    pub fn predict<F>(&mut self, dt: f64, mut transition_fn: F) -> Result<(), FilterError>
    where
        F: FnMut([f64; 2], f64) -> [f64; 2],
    {
        let (wm, wc, sigma_points) = self.generate_sigma_points()?;

        // Propagate sigma points through the non-linear dynamics
        let mut propagated_points = [[0.0; 2]; 5];
        for i in 0..5 {
            propagated_points[i] = transition_fn(sigma_points[i], dt);
            if !propagated_points[i][0].is_finite() || !propagated_points[i][1].is_finite() {
                return Err(FilterError::NonFiniteInput);
            }
        }

        // Compute predicted mean: x_pred = sum(wm[i] * chi[i])
        let mut x_pred = [0.0; 2];
        for i in 0..5 {
            x_pred[0] += wm[i] * propagated_points[i][0];
            x_pred[1] += wm[i] * propagated_points[i][1];
        }

        // Compute predicted covariance: P_pred = sum(wc[i] * (chi[i] - x_pred) * (chi[i] - x_pred)^T) + Q
        let mut p_pred = [[0.0; 2]; 2];
        for i in 0..5 {
            let dx0 = propagated_points[i][0] - x_pred[0];
            let dx1 = propagated_points[i][1] - x_pred[1];

            p_pred[0][0] += wc[i] * dx0 * dx0;
            p_pred[0][1] += wc[i] * dx0 * dx1;
            p_pred[1][0] += wc[i] * dx1 * dx0;
            p_pred[1][1] += wc[i] * dx1 * dx1;
        }

        p_pred[0][0] += self.q[0][0];
        p_pred[0][1] += self.q[0][1];
        p_pred[1][0] += self.q[1][0];
        p_pred[1][1] += self.q[1][1];

        // Symmetrize
        let p_sym = (p_pred[0][1] + p_pred[1][0]) * 0.5;
        p_pred[0][1] = p_sym;
        p_pred[1][0] = p_sym;

        self.x = x_pred;
        self.p = p_pred;
        Ok(())
    }

    /// Measurement update through an arbitrary non-linear observation function `h(x) -> scalar`.
    pub fn update<H>(&mut self, measurement: f64, mut observe_fn: H, gate_sigma: Option<f64>) -> Result<[f64; 2], FilterError>
    where
        H: FnMut([f64; 2]) -> f64,
    {
        if !measurement.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }

        let (wm, wc, sigma_points) = self.generate_sigma_points()?;

        // Map sigma points into measurement space
        let mut gamma = [0.0; 5];
        for i in 0..5 {
            gamma[i] = observe_fn(sigma_points[i]);
            if !gamma[i].is_finite() {
                return Err(FilterError::NonFiniteInput);
            }
        }

        // Predicted measurement: z_pred = sum(wm[i] * gamma[i])
        let mut z_pred = 0.0;
        for i in 0..5 {
            z_pred += wm[i] * gamma[i];
        }

        // Innovation covariance: S = sum(wc[i] * (gamma[i] - z_pred)^2) + R
        let mut s = self.r;
        for i in 0..5 {
            let dz = gamma[i] - z_pred;
            s += wc[i] * dz * dz;
        }

        if s <= 1e-15 || !s.is_finite() {
            return Err(FilterError::SingularInnovation);
        }

        // Innovation residual
        let y = measurement - z_pred;

        // Outlier rejection
        if let Some(gate) = gate_sigma {
            if gate > 0.0 {
                let normalized_err = y.abs_val() / s.sqrt_val();
                if normalized_err > gate {
                    return Err(FilterError::MeasurementGated);
                }
            }
        }

        // Cross-covariance: P_xz = sum(wc[i] * (chi[i] - x) * (gamma[i] - z_pred))
        let mut p_xz = [0.0; 2];
        for i in 0..5 {
            let dx0 = sigma_points[i][0] - self.x[0];
            let dx1 = sigma_points[i][1] - self.x[1];
            let dz = gamma[i] - z_pred;

            p_xz[0] += wc[i] * dx0 * dz;
            p_xz[1] += wc[i] * dx1 * dz;
        }

        // Kalman Gain: K = P_xz / S
        let k0 = p_xz[0] / s;
        let k1 = p_xz[1] / s;

        // State update
        self.x[0] += k0 * y;
        self.x[1] += k1 * y;

        // Covariance update: P = P - K * S * K^T
        self.p[0][0] -= k0 * k0 * s;
        self.p[0][1] -= k0 * k1 * s;
        self.p[1][0] -= k1 * k0 * s;
        self.p[1][1] -= k1 * k1 * s;

        // Ensure floor and symmetry
        let p_sym = (self.p[0][1] + self.p[1][0]) * 0.5;
        self.p[0][1] = p_sym;
        self.p[1][0] = p_sym;
        if self.p[0][0] < 1e-15 {
            self.p[0][0] = 1e-15;
        }
        if self.p[1][1] < 1e-15 {
            self.p[1][1] = 1e-15;
        }

        Ok(self.x)
    }
}
