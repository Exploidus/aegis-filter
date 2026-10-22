//! Robust & Adaptive State Estimation.
//!
//! Features:
//! - **$H_\infty$ (H-infinity) Bounded-Error Minimax Filter**: Minimizes worst-case estimation error
//!   regardless of noise probability distribution, making it impervious to non-Gaussian
//!   sensor anomalies, unmodeled dynamics, and adversarial attacks.
//! - **Adaptive Innovation Noise Estimator**: Automatically tunes measurement noise ($R$)
//!   at runtime using residual variance matching (Sage-Husa formulation).
//!
//! # References
//! - Dan Simon, *Optimal State Estimation: Kalman, H-infinity, and Nonlinear Approaches*,
//!   John Wiley & Sons, 2006, Chapter 11: "H-infinity Filtering", pp. 343-355.
//! - Babak Hassibi, Ali H. Sayed, and Thomas Kailath, *Indefinite-Quadratic Estimation and Filtering: Theory and Applications*,
//!   SIAM Studies in Applied Mathematics, Vol. 16, 1999.
//! - Xiang Shen and Li Deng, "Game theory approach to discrete H-infinity filter design",
//!   *IEEE Transactions on Signal Processing*, 45(4):1092-1095, 1997.

use crate::common::FilterError;

/// 1D Robust $H_\infty$ (H-infinity) Minimax Filter.
///
/// Unlike the classical Kalman Filter, which minimizes the *mean-square error* under the
/// assumption of known Gaussian white noise, the $H_\infty$ filter solves a minimax game:
///
/// $$J = \frac{\sum_{k=0}^{N-1} \| z_k - \hat{z}_k \|_{S_k}^2}{\| x_0 - \hat{x}_0 \|_{P_0^{-1}}^2 + \sum_{k=0}^{N-1} (\| w_k \|_{Q_k^{-1}}^2 + \| v_k \|_{R_k^{-1}}^2)} < \gamma^2$$
///
/// It bounds the maximum energy gain from disturbance noise to estimation error by $\gamma^2$.
///
/// # Mathematical Formulation (Discrete-Time Scalar Case)
/// Following Simon (2006, Eq. 11.45-11.48) for scalar state, direct observation ($H = 1$),
/// and state error weighting ($L = 1$):
///
/// 1. **Time Update**:
///    $$x_{k|k-1} = x_{k-1|k-1} + u_k$$
///    $$P_{k|k-1} = P_{k-1|k-1} + Q$$
///
/// 2. **Riccati Denominator Operator**:
///    $$D = 1 - \gamma^{-2} P_{k|k-1} + R^{-1} P_{k|k-1}$$
///
/// 3. **Bounded-Real Stability Condition**:
///    For a valid $H_\infty$ filter to exist, the Riccati operator must remain positive:
///    $$D > 0 \iff P_{k|k-1}^{-1} - \gamma^{-2} + R^{-1} > 0$$
///    If $D \le 0$, the guaranteed attenuation bound $\gamma$ is violated and the filter returns
///    [`FilterError::NumericalInstability`].
///
/// 4. **Gain and Measurement Update**:
///    $$K_k = \frac{P_{k|k-1}}{R \cdot D}$$
///    $$x_{k|k} = x_{k|k-1} + K_k (z_k - x_{k|k-1})$$
///    $$P_{k|k} = \frac{P_{k|k-1}}{D}$$
///
/// # Example
/// ```rust
/// use aegis_filter::HInfinityFilter1D;
///
/// let mut filter = HInfinityFilter1D::new(0.0, 1.0, 0.01, 0.5, 5.0).unwrap();
/// filter.predict(0.1).unwrap();
/// let est = filter.update(1.05).unwrap();
/// assert!(est > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HInfinityFilter1D {
    /// State estimate ($x$)
    x: f64,
    /// Covariance / error bound parameter ($P$)
    p: f64,
    /// Process disturbance bound ($Q$)
    q: f64,
    /// Measurement disturbance bound ($R$)
    r: f64,
    /// Performance attenuation bound ($\gamma$). Smaller values enforce tighter worst-case error bounds.
    gamma: f64,
}

impl HInfinityFilter1D {
    /// Creates a new $H_\infty$ filter.
    /// `gamma` must be strictly greater than 0 (typically $\ge 1.0$).
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
    /// $$x = x + u$$
    /// $$P = P + Q$$
    pub fn predict(&mut self, control_input: f64) -> Result<(), FilterError> {
        if !control_input.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }
        self.x += control_input;
        self.p += self.q;
        Ok(())
    }

    /// Measurement update under the $H_\infty$ minimax criteria:
    ///
    /// Stability condition requires positive-definiteness of Riccati operator:
    /// $$D = 1 - \frac{P}{\gamma^2} + \frac{P}{R} > 0$$
    pub fn update(&mut self, measurement: f64) -> Result<f64, FilterError> {
        if !measurement.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }

        let gamma_sq = self.gamma * self.gamma;

        // Bounded-real Riccati denominator term (Simon 2006, Eq. 11.45):
        // D = 1 - (1.0 / gamma^2) * P + (H^2 * P) / R  where H = 1.0
        let d = 1.0 - (self.p / gamma_sq) + (self.p / self.r);

        // If the H-infinity bound is violated, the minimax bound cannot be guaranteed
        if d <= 1e-12 || !d.is_finite() {
            return Err(FilterError::NumericalInstability);
        }

        // H-infinity Gain (Simon 2006, Eq. 11.46):
        // K = P / (R * D)
        let k = self.p / (self.r * d);

        // State update
        let residual = measurement - self.x;
        self.x += k * residual;

        // Covariance update: P = P / D (Simon 2006, Eq. 11.47)
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

    #[inline(always)]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

/// Adaptive Innovation Noise Estimator (Sage-Husa Formulation).
///
/// Automatically adjusts measurement variance $R$ online using an exponential moving average
/// of the pre-update innovation sequence.
///
/// # Mathematical Principle
/// For innovation $y_k = z_k - x_{k|k-1}$, the theoretical innovation variance is:
/// $$E[y_k^2] = P_{k|k-1} + R_k$$
///
/// The instantaneous unbiased sample estimate of measurement noise is therefore:
/// $$\hat{R}_k = y_k^2 - P_{k|k-1}$$
///
/// Notice this MUST be computed with the *pre-update* predicted covariance $P_{k|k-1}$,
/// before the posterior Joseph covariance update modifies $P$. The estimator clamps
/// $\hat{R}_k$ to $[10^{-6}, 10^6]$ to prevent negative variance or unbounded upward drift.
///
/// # Example
/// ```rust
/// use aegis_filter::AdaptiveFilter1D;
///
/// let mut af = AdaptiveFilter1D::new(0.0, 1.0, 0.01, 0.5, 0.05).unwrap();
/// af.predict(0.0).unwrap();
/// let est = af.update(5.0).unwrap();
/// assert!(est > 0.0);
/// assert!(af.estimated_measurement_noise() > 0.0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AdaptiveFilter1D {
    pub x: f64,
    pub p: f64,
    pub q: f64,
    pub r: f64,
    /// Smoothing factor for online noise estimation (0.01 to 0.1)
    pub alpha: f64,
}

impl AdaptiveFilter1D {
    pub fn new(
        initial_x: f64,
        initial_p: f64,
        q: f64,
        initial_r: f64,
        adaptation_rate: f64,
    ) -> Result<Self, FilterError> {
        if !initial_x.is_finite()
            || initial_p <= 0.0
            || q < 0.0
            || initial_r <= 0.0
            || adaptation_rate <= 0.0
            || adaptation_rate >= 1.0
        {
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

    pub fn predict(&mut self, u: f64) -> Result<(), FilterError> {
        if !u.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }
        self.x += u;
        self.p += self.q;
        Ok(())
    }

    pub fn update(&mut self, z: f64) -> Result<f64, FilterError> {
        if !z.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }

        // Store pre-update predicted covariance for Sage-Husa noise estimation
        let p_prior = self.p;
        let y = z - self.x;
        let s = p_prior + self.r;

        if s <= 0.0 || !s.is_finite() {
            return Err(FilterError::SingularInnovation);
        }

        let k = p_prior / s;
        self.x += k * y;

        // Joseph form update using prior covariance: P = (1-K)^2 * P_prior + K^2 * R
        let one_minus_k = 1.0 - k;
        self.p = (one_minus_k * one_minus_k * p_prior) + (k * k * self.r);

        // Sage-Husa adaptive estimation of R:
        // Instantaneous measurement noise estimate: R_sample = y^2 - P_prior
        let r_sample = y * y - p_prior;
        let r_sample_bounded = r_sample.clamp(1e-6, 1e6);

        // Exponential moving average update
        self.r = (1.0 - self.alpha) * self.r + self.alpha * r_sample_bounded;
        self.r = self.r.clamp(1e-6, 1e6);

        Ok(self.x)
    }

    #[inline(always)]
    pub fn estimated_measurement_noise(&self) -> f64 {
        self.r
    }
}
