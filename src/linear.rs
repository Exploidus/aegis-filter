//! Classical Linear & Kinematic Kalman Filters with Joseph stabilized covariance.

use crate::common::{FilterError, FloatOps, Real};

/// 1D Scalar Kalman Filter with Joseph-stabilized variance update.
///
/// # Example
/// ```rust
/// use aegis_filter::ScalarKalmanFilter;
///
/// let mut filter = ScalarKalmanFilter::new(0.0, 1.0, 0.01, 0.5).unwrap();
/// filter.predict(0.1).unwrap();
/// let est = filter.update(1.05, Some(3.0)).unwrap();
/// assert!(est > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScalarKalmanFilter {
    x: Real,
    p: Real,
    q: Real,
    r: Real,
    default_gate: Option<Real>,
}

impl ScalarKalmanFilter {
    #[allow(clippy::needless_range_loop)]
    pub fn new(
        initial_state: Real,
        initial_variance: Real,
        process_noise: Real,
        measurement_noise: Real,
    ) -> Result<Self, FilterError> {
        if !initial_state.is_finite()
            || !initial_variance.is_finite()
            || !process_noise.is_finite()
            || !measurement_noise.is_finite()
        {
            return Err(FilterError::NonFiniteInput);
        }

        if initial_variance <= 0.0 || process_noise < 0.0 || measurement_noise <= 0.0 {
            return Err(FilterError::InvalidCovariance);
        }

        Ok(Self {
            x: initial_state,
            p: initial_variance,
            q: process_noise,
            r: measurement_noise,
            default_gate: None,
        })
    }

    /// Creates a fail-closed, cyber-resilient filter with default 3-sigma innovation gating enabled.
    pub fn new_resilient(
        initial_state: Real,
        initial_variance: Real,
        process_noise: Real,
        measurement_noise: Real,
    ) -> Result<Self, FilterError> {
        let mut filter = Self::new(
            initial_state,
            initial_variance,
            process_noise,
            measurement_noise,
        )?;
        filter.default_gate = Some(crate::DEFAULT_OUTLIER_GATE_SIGMA);
        Ok(filter)
    }

    /// Sets or overrides the persistent default gating threshold.
    ///
    /// When configured, calls to [`Self::update`] with `None` will enforce this gate (fail-closed).
    pub fn with_default_gate(mut self, threshold_sigma: Real) -> Result<Self, FilterError> {
        if !threshold_sigma.is_finite() || threshold_sigma <= 0.0 {
            return Err(FilterError::InvalidParameter);
        }
        self.default_gate = Some(threshold_sigma);
        Ok(self)
    }

    /// Disables default innovation gating, returning to unconstrained (fail-open) mode.
    pub fn disable_default_gate(mut self) -> Self {
        self.default_gate = None;
        self
    }

    pub fn predict(&mut self, control_input: Real) -> Result<(), FilterError> {
        if !control_input.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }

        self.x += control_input;
        self.p += self.q;

        if !self.x.is_finite() || !self.p.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }

        Ok(())
    }

    /// Fuses an incoming scalar measurement with optional innovation outlier gating.
    ///
    /// # Gating Semantics
    /// - `Some(sigma)`: Overrides gating for this call: rejects measurements whose normalized
    ///   innovation $|y| / \sqrt{S} > \sigma$ with [`FilterError::MeasurementGated`].
    /// - `Some(<= 0.0)`: Explicitly bypasses outlier gating for this update.
    /// - `None`: Falls back to the filter's persistent default gate ([`Self::with_default_gate`] or
    ///   [`Self::new_resilient`]). If no default was configured, gating is bypassed.
    pub fn update(
        &mut self,
        measurement: Real,
        gate_threshold_sigma: Option<Real>,
    ) -> Result<Real, FilterError> {
        if !measurement.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }

        let y = measurement - self.x;
        let s = self.p + self.r;
        if s <= 1e-15 || !s.is_finite() {
            return Err(FilterError::SingularInnovation);
        }

        let effective_gate = match gate_threshold_sigma {
            Some(sigma) if sigma <= 0.0 => None,
            Some(sigma) => Some(sigma),
            None => self.default_gate,
        };

        if let Some(gate_sigma) = effective_gate {
            let normalized_residual = y.abs_val() / s.sqrt_val();
            if normalized_residual > gate_sigma {
                return Err(FilterError::MeasurementGated);
            }
        }

        let k = self.p / s;
        self.x += k * y;

        // Joseph stabilized update: P = (1 - K)^2 * P + K^2 * R
        let one_minus_k = 1.0 - k;
        self.p = (one_minus_k * one_minus_k * self.p) + (k * k * self.r);

        if self.p < 1e-15 {
            self.p = 1e-15;
        }

        Ok(self.x)
    }

    #[inline(always)]
    pub fn state(&self) -> Real {
        self.x
    }

    #[inline(always)]
    pub fn variance(&self) -> Real {
        self.p
    }
}

/// Generic N-dimensional Linear Kalman Filter using const-generics.
#[derive(Debug, Clone)]
pub struct LinearKalmanFilter<const S: usize, const M: usize> {
    pub x: [Real; S],
    pub p: [[Real; S]; S],
    pub f: [[Real; S]; S],
    pub q: [[Real; S]; S],
    pub h: [[Real; S]; M],
    pub r: [[Real; M]; M],
    pub default_gate: Option<Real>,
}

impl<const S: usize, const M: usize> LinearKalmanFilter<S, M> {
    #[allow(clippy::needless_range_loop)]
    pub fn new(
        initial_x: [Real; S],
        initial_p: [[Real; S]; S],
        f: [[Real; S]; S],
        q: [[Real; S]; S],
        h: [[Real; S]; M],
        r: [[Real; M]; M],
    ) -> Result<Self, FilterError> {
        for val in initial_x.iter() {
            if !val.is_finite() {
                return Err(FilterError::NonFiniteInput);
            }
        }
        Self::validate_matrix(&initial_p)?;
        Self::validate_matrix(&f)?;
        Self::validate_matrix(&q)?;
        Self::validate_matrix(&h)?;
        Self::validate_matrix(&r)?;

        // Physical covariance validation: diagonal variances must be non-negative,
        // and initial state uncertainty P must be strictly positive.
        for i in 0..S {
            if initial_p[i][i] <= 0.0 {
                return Err(FilterError::InvalidCovariance);
            }
            if q[i][i] < 0.0 {
                return Err(FilterError::InvalidCovariance);
            }
        }
        for j in 0..M {
            if r[j][j] < 0.0 {
                return Err(FilterError::InvalidCovariance);
            }
        }

        Ok(Self {
            x: initial_x,
            p: initial_p,
            f,
            q,
            h,
            r,
            default_gate: None,
        })
    }

    /// Creates a fail-closed, cyber-resilient filter with default 3-sigma Mahalanobis gating enabled.
    pub fn new_resilient(
        initial_x: [Real; S],
        initial_p: [[Real; S]; S],
        f: [[Real; S]; S],
        q: [[Real; S]; S],
        h: [[Real; S]; M],
        r: [[Real; M]; M],
    ) -> Result<Self, FilterError> {
        let mut filter = Self::new(initial_x, initial_p, f, q, h, r)?;
        filter.default_gate = Some(crate::DEFAULT_OUTLIER_GATE_SIGMA);
        Ok(filter)
    }

    /// Sets or overrides the persistent default gating threshold.
    ///
    /// When configured, calls to [`Self::update`] with `None` will enforce this gate (fail-closed).
    pub fn with_default_gate(mut self, threshold_sigma: Real) -> Result<Self, FilterError> {
        if !threshold_sigma.is_finite() || threshold_sigma <= 0.0 {
            return Err(FilterError::InvalidParameter);
        }
        self.default_gate = Some(threshold_sigma);
        Ok(self)
    }

    /// Disables default Mahalanobis gating, returning to unconstrained (fail-open) mode.
    pub fn disable_default_gate(mut self) -> Self {
        self.default_gate = None;
        self
    }

    fn validate_matrix<const R: usize, const C: usize>(
        matrix: &[[Real; C]; R],
    ) -> Result<(), FilterError> {
        for row in matrix.iter() {
            for val in row.iter() {
                if !val.is_finite() {
                    return Err(FilterError::NonFiniteInput);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    pub fn predict(&mut self) -> Result<(), FilterError> {
        let mut x_pred = [0.0; S];
        for i in 0..S {
            let mut sum = 0.0;
            for j in 0..S {
                sum += self.f[i][j] * self.x[j];
            }
            x_pred[i] = sum;
        }

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

        let mut p_pred = [[0.0; S]; S];
        for i in 0..S {
            for j in 0..S {
                let mut sum = 0.0;
                for k in 0..S {
                    sum += f_p[i][k] * self.f[j][k];
                }
                p_pred[i][j] = sum + self.q[i][j];
            }
        }

        for i in 0..S {
            for j in (i + 1)..S {
                let avg = (p_pred[i][j] + p_pred[j][i]) * 0.5;
                p_pred[i][j] = avg;
                p_pred[j][i] = avg;
            }
        }

        for i in 0..S {
            if !x_pred[i].is_finite() {
                return Err(FilterError::NonFiniteInput);
            }
            for j in 0..S {
                if !p_pred[i][j].is_finite() {
                    return Err(FilterError::NonFiniteInput);
                }
            }
        }

        self.x = x_pred;
        self.p = p_pred;
        Ok(())
    }

    /// Fuse a length-`M` measurement `z` into the state estimate.
    ///
    /// This generalises the 1D [`ScalarKalmanFilter::update`] to the multi-dimensional
    /// case, keeping the same guarantees:
    ///
    /// * **Joseph stabilised covariance update.** After `P = (I - K H) P (I - K H)^T + K R K^T`
    ///   the posterior is symmetric by construction and stays positive semi-definite under
    ///   finite round-off, which the naive `P - K S K^T` subtraction form cannot promise.
    /// * **Gating.** With `gate_threshold_sigma` set, the measurement is rejected as an outlier
    ///   when its Mahalanobis distance `sqrt(y^T S^-1 y)` exceeds the threshold (the state is
    ///   left unchanged). For `M = 1` this reduces exactly to the 1D `|y| / sqrt(S)` gate.
    /// * **No unwinding.** Every failure path is a [`FilterError`], never a panic; a singular
    ///   innovation covariance returns [`FilterError::NumericalInstability`].
    ///
    /// # Example
    /// ```rust
    /// use aegis_filter::LinearKalmanFilter;
    ///
    /// // Single measurement of a 1-D state (H = I, R = 0.5).
    /// let mut f = LinearKalmanFilter::<1, 1>::new(
    ///     [0.0], [[1.0]], [[1.0]], [[0.01]], [[1.0]], [[0.5]]
    /// ).unwrap();
    /// f.predict().unwrap();
    /// let est = f.update(&[1.05], None).unwrap();
    /// assert!(est[0] > 0.0);
    /// ```
    /// Fuses an incoming measurement vector with optional Mahalanobis outlier gating.
    ///
    /// # Gating Semantics (3-Sigma vs Mahalanobis Distance)
    /// - `Some(sigma)`: Rejects measurements whose Mahalanobis distance $\sqrt{y^T S^{-1} y} > \sigma$
    ///   with [`FilterError::MeasurementGated`], leaving the state and covariance unmodified.
    ///   Use [`crate::DEFAULT_OUTLIER_GATE_SIGMA`] (3.0) for standard multi-sigma protection.
    /// - `None` or `Some(<= 0.0)`: Outlier gating is bypassed (fail-open mode).
    ///
    /// For 1D scalar observation ($M = 1$), the Mahalanobis distance simplifies directly to
    /// the 1D normalized residual $|y| / \sqrt{S}$. For multi-dimensional observations ($M > 1$),
    /// the metric accounts for correlated sensor cross-covariances through innovation matrix inversion $S^{-1}$.
    #[allow(clippy::needless_range_loop)] // manual index loops over const-generic buffers
    pub fn update(
        &mut self,
        measurement: &[Real; M],
        gate_threshold_sigma: Option<Real>,
    ) -> Result<[Real; S], FilterError> {
        // Boundary check: non-finite measurements are rejected, like the 1D path.
        for &val in measurement.iter() {
            if !val.is_finite() {
                return Err(FilterError::NonFiniteInput);
            }
        }

        // Snapshot the pre-update state and covariance; the Joseph form must read P old.
        let x = self.x;
        let p = self.p;
        let h = self.h;
        let r = self.r;

        // Innovation y = z - H x.
        let mut y = [0.0; M];
        for j in 0..M {
            let mut z_hat = 0.0;
            for i in 0..S {
                z_hat += h[j][i] * x[i];
            }
            y[j] = measurement[j] - z_hat;
        }

        // Innovation covariance S = H P H^T + R, then symmetrise it.
        // Factoring H P H^T via intermediate HP = H * P:
        // 1. HP = H * P (M x S) in O(M * S^2).
        // Since P is symmetric (P = P^T), (H * P)^T = P * H^T.
        let mut hp = [[0.0; S]; M];
        for j in 0..M {
            for i in 0..S {
                let mut acc = 0.0;
                for l in 0..S {
                    acc += h[j][l] * p[l][i];
                }
                hp[j][i] = acc;
            }
        }

        // 2. S = (HP) * H^T + R (M x M) in O(M^2 * S).
        let mut s = [[0.0; M]; M];
        for j in 0..M {
            for k in 0..M {
                let mut acc = 0.0;
                for i in 0..S {
                    acc += hp[j][i] * h[k][i];
                }
                s[j][k] = acc + r[j][k];
            }
        }
        for j in 0..M {
            for k in (j + 1)..M {
                let avg = (s[j][k] + s[k][j]) * 0.5;
                s[j][k] = avg;
                s[k][j] = avg;
            }
        }

        // Invert S. A singular or ill-conditioned innovation covariance is a hard
        // failure the caller can react to, rather than a panic.
        let s_inv = Self::invert_matrix(&s)?;

        // Gating: Mahalanobis distance sqrt(y^T S^-1 y). For M = 1 this is |y|/sqrt(S),
        // identical to the 1D gate.
        let effective_gate = match gate_threshold_sigma {
            Some(sigma) if sigma <= 0.0 => None,
            Some(sigma) => Some(sigma),
            None => self.default_gate,
        };

        if let Some(gate_sigma) = effective_gate {
            let mut stat = 0.0;
            for j in 0..M {
                for k in 0..M {
                    stat += y[j] * s_inv[j][k] * y[k];
                }
            }
            // stat is theoretically >= 0; guard round-off that could dip below zero.
            let mahal = stat.max(0.0).sqrt_val();
            if mahal > gate_sigma {
                return Err(FilterError::MeasurementGated);
            }
        }

        // Kalman gain K = P H^T S^-1  (S x M), computed using the factored HP transpose.
        // 1. A = P H^T  (S x M) is the transpose of HP: A[i][l] = hp[l][i].
        let mut p_ht = [[0.0; M]; S];
        for i in 0..S {
            for l in 0..M {
                p_ht[i][l] = hp[l][i];
            }
        }
        // 2. K = A S^-1  (S x M):  K[i][j] = sum_l P_H_T[i][l] S^-1[l][j]
        let mut k = [[0.0; M]; S];
        for i in 0..S {
            for j in 0..M {
                let mut acc = 0.0;
                for l in 0..M {
                    acc += p_ht[i][l] * s_inv[l][j];
                }
                k[i][j] = acc;
            }
        }

        // State update x = x + K y.
        let mut x_new = x;
        for i in 0..S {
            let mut acc = 0.0;
            for j in 0..M {
                acc += k[i][j] * y[j];
            }
            x_new[i] = x[i] + acc;
        }

        // Joseph covariance update: P = (I - K H) P (I - K H)^T + K R K^T.
        // I - K H is S x S.
        let mut e = [[0.0; S]; S];
        for i in 0..S {
            for j in 0..S {
                let mut k_h = 0.0;
                for t in 0..M {
                    k_h += k[i][t] * h[t][j];
                }
                e[i][j] = if i == j { 1.0 - k_h } else { -k_h };
            }
        }

        // Factored Joseph form: O(S^3 + S^2 M + S M^2) instead of fused O(S^4 + S^2 M^2).
        // 1. EP = E * P  (S x S) in O(S^3)
        let mut ep = [[0.0; S]; S];
        for i in 0..S {
            for j in 0..S {
                let mut acc = 0.0;
                for a in 0..S {
                    acc += e[i][a] * p[a][j];
                }
                ep[i][j] = acc;
            }
        }

        // 2. KR = K * R  (S x M) in O(S * M^2)
        let mut kr = [[0.0; M]; S];
        for i in 0..S {
            for t in 0..M {
                let mut acc = 0.0;
                for l in 0..M {
                    acc += k[i][l] * r[l][t];
                }
                kr[i][t] = acc;
            }
        }

        // 3. P_new = (EP) * E^T + (KR) * K^T in O(S^2 * (S + M))
        let mut p_new = [[0.0; S]; S];
        for i in 0..S {
            for j in 0..S {
                let mut ep_e = 0.0;
                for b in 0..S {
                    ep_e += ep[i][b] * e[j][b];
                }
                let mut k_r_k = 0.0;
                for t in 0..M {
                    k_r_k += kr[i][t] * k[j][t];
                }
                p_new[i][j] = ep_e + k_r_k;
            }
        }

        // Symmetrise and floor the diagonals: the Joseph form can leave a tiny negative
        // on the diagonal under round-off; flooring keeps P well-posed for the next step.
        for i in 0..S {
            for j in i..S {
                let avg = (p_new[i][j] + p_new[j][i]) * 0.5;
                p_new[i][j] = avg;
                p_new[j][i] = avg;
            }
            if p_new[i][i] < 1e-15 {
                p_new[i][i] = 1e-15;
            }
        }

        // Final guard: the update must not emit non-finite state or covariance.
        if !x_new.iter().all(|v| v.is_finite()) || !p_new.iter().flatten().all(|v| v.is_finite()) {
            return Err(FilterError::NumericalInstability);
        }

        self.x = x_new;
        self.p = p_new;
        Ok(x_new)
    }

    /// Gauss-Jordan inversion of a const-sized `N x N` matrix with partial pivoting.
    ///
    /// Invert is the one numerically sensitive step in [`update`]; a ~zero pivot after
    /// pivoting means the matrix is singular (or numerically rank-deficient) and we return
    /// [`FilterError::NumericalInstability`] rather than divide by it. Threshold `1e-15`
    /// matches the singularity guard the 1D filter uses for its scalar innovation covariance.
    #[allow(clippy::needless_range_loop)] // index loops over const-generic buffers
    fn invert_matrix<const N: usize>(m: &[[Real; N]; N]) -> Result<[[Real; N]; N], FilterError> {
        // Two N x N buffers instead of one 2*N augmented block: a generic const param may
        // not appear in const arithmetic (`2 * N`) under edition 2021. `mat` accumulates the
        // pivots so its left side becomes I; `inv` follows the identical row ops and so
        // becomes M^-1. Both stay on the stack, keeping the crate zero-allocation / no_std.
        let mut mat = [[0.0; N]; N];
        let mut inv = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                mat[i][j] = m[i][j];
                inv[i][j] = if i == j { 1.0 } else { 0.0 };
            }
        }

        for col in 0..N {
            // Partial pivot: pick the row with the largest |pivot| in this column to
            // avoid dividing by a near-zero value (improves numerical stability).
            let mut pivot = col;
            for r in (col + 1)..N {
                if mat[r][col].abs_val() > mat[pivot][col].abs_val() {
                    pivot = r;
                }
            }
            if mat[pivot][col].abs_val() < 1e-15 {
                return Err(FilterError::NumericalInstability);
            }
            // The pivot swap must hit both buffers so they stay row-aligned.
            mat.swap(col, pivot);
            inv.swap(col, pivot);

            // Normalise the pivot row to a unit diagonal.
            let inv_diag = 1.0 / mat[col][col];
            for j in 0..N {
                mat[col][j] *= inv_diag;
                inv[col][j] *= inv_diag;
            }

            // Eliminate this column from every other row, applying the same op to `inv`.
            for r in 0..N {
                if r == col {
                    continue;
                }
                let factor = mat[r][col];
                if factor == 0.0 {
                    continue;
                }
                for j in 0..N {
                    mat[r][j] -= factor * mat[col][j];
                    inv[r][j] -= factor * inv[col][j];
                }
            }
        }

        // When mat is the identity, inv holds M^-1.
        Ok(inv)
    }
}
