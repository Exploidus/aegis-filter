//! Classical Linear & Kinematic Kalman Filters with Joseph stabilized covariance.

use crate::common::{FilterError, FloatOps, Real, DEFAULT_MAX_CONDITION_NUMBER};

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
        crate::simd::matmul::<S, S, S>(&self.f, &self.p, &mut f_p);

        let mut p_pred = [[0.0; S]; S];
        crate::simd::matmul_transpose_b::<S, S, S>(&f_p, &self.f, &mut p_pred);

        for i in 0..S {
            for j in 0..S {
                p_pred[i][j] += self.q[i][j];
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

        // SIMD accelerated Joseph form update: O(S^3 + S^2 M + S M^2)
        let mut p_new = [[0.0; S]; S];
        crate::simd::joseph_form_covariance_update::<S, M>(&e, &p, &k, &r, &mut p_new);

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

    /// Computes the 1-norm numerical condition number $\kappa_1(P) = \|P\|_1 \cdot \|P^{-1}\|_1$ of state covariance $P$.
    ///
    /// An ill-conditioned covariance matrix ($\kappa \gg 1$) indicates loss of numerical precision or near-singularity,
    /// which can cause filter divergence during Kalman gain computation.
    #[inline]
    pub fn condition_number(&self) -> Result<Real, FilterError> {
        matrix_condition_number(&self.p)
    }

    /// Checks whether state covariance $P$ is well-conditioned against a specified maximum threshold.
    ///
    /// Rejects non-positive, asymmetric, or ill-conditioned covariances with [`FilterError::InvalidCovariance`].
    #[inline]
    pub fn check_conditioning(&self, max_condition_number: Real) -> Result<Real, FilterError> {
        check_matrix_conditioning(&self.p, max_condition_number)
    }

    /// Quick boolean check whether state covariance $P$ is well-conditioned against [`DEFAULT_MAX_CONDITION_NUMBER`].
    #[inline]
    pub fn is_well_conditioned(&self) -> bool {
        self.check_conditioning(DEFAULT_MAX_CONDITION_NUMBER)
            .is_ok()
    }

    /// Inverts an $N \times N$ matrix via Gauss-Jordan elimination with partial pivoting.
    ///
    /// Kept for backwards compatibility; delegates to [`invert_matrix`].
    #[inline(always)]
    pub fn invert_matrix<const N: usize>(
        m: &[[Real; N]; N],
    ) -> Result<[[Real; N]; N], FilterError> {
        invert_matrix(m)
    }
}

#[cfg(not(feature = "single-precision"))]
const SINGULARITY_EPSILON: Real = 1e-15;

#[cfg(feature = "single-precision")]
const SINGULARITY_EPSILON: Real = 1e-7;

/// Computes the matrix 1-norm (maximum absolute column sum) in $O(R \cdot C)$ with zero heap allocation.
///
/// $$\|M\|_1 = \max_{0 \le j < C} \sum_{i=0}^{R-1} |M_{ij}|$$
#[allow(clippy::needless_range_loop)]
pub fn matrix_norm_1<const R: usize, const C: usize>(m: &[[Real; C]; R]) -> Real {
    let mut max_col_sum: Real = 0.0;
    for col in 0..C {
        let mut col_sum: Real = 0.0;
        for row in 0..R {
            col_sum += m[row][col].abs_val();
        }
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }
    max_col_sum
}

/// Computes the matrix $\infty$-norm (maximum absolute row sum) in $O(R \cdot C)$ with zero heap allocation.
///
/// $$\|M\|_\infty = \max_{0 \le i < R} \sum_{j=0}^{C-1} |M_{ij}|$$
#[allow(clippy::needless_range_loop)]
pub fn matrix_norm_inf<const R: usize, const C: usize>(m: &[[Real; C]; R]) -> Real {
    let mut max_row_sum: Real = 0.0;
    for row in 0..R {
        let mut row_sum: Real = 0.0;
        for col in 0..C {
            row_sum += m[row][col].abs_val();
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }
    max_row_sum
}

/// Computes the matrix Frobenius norm in $O(R \cdot C)$ with zero heap allocation.
///
/// $$\|M\|_F = \sqrt{\sum_{i=0}^{R-1} \sum_{j=0}^{C-1} M_{ij}^2}$$
#[allow(clippy::needless_range_loop)]
pub fn matrix_norm_frobenius<const R: usize, const C: usize>(m: &[[Real; C]; R]) -> Real {
    let mut sum_sq: Real = 0.0;
    for row in 0..R {
        for col in 0..C {
            let val = m[row][col];
            sum_sq += val * val;
        }
    }
    sum_sq.sqrt_val()
}

/// Gauss-Jordan inversion of a const-sized $N \times N$ matrix with partial pivoting in pure `#[no_std]` stack memory.
///
/// Inversion is performed in-place using two stack-allocated buffers `mat` and `inv`, ensuring
/// zero heap allocation. Returns [`FilterError::NumericalInstability`] if a pivot is near-zero (singular matrix).
#[allow(clippy::needless_range_loop)]
pub fn invert_matrix<const N: usize>(m: &[[Real; N]; N]) -> Result<[[Real; N]; N], FilterError> {
    let mut mat = [[0.0; N]; N];
    let mut inv = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            mat[i][j] = m[i][j];
            inv[i][j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    for col in 0..N {
        let mut pivot = col;
        for r in (col + 1)..N {
            if mat[r][col].abs_val() > mat[pivot][col].abs_val() {
                pivot = r;
            }
        }
        if mat[pivot][col].abs_val() < SINGULARITY_EPSILON {
            return Err(FilterError::NumericalInstability);
        }
        mat.swap(col, pivot);
        inv.swap(col, pivot);

        let inv_diag = 1.0 / mat[col][col];
        for j in 0..N {
            mat[col][j] *= inv_diag;
            inv[col][j] *= inv_diag;
        }

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

    Ok(inv)
}

/// Computes the exact 1-norm condition number $\kappa_1(M) = \|M\|_1 \cdot \|M^{-1}\|_1$ for square matrices.
///
/// Returns:
/// - `Ok(kappa)` where $\kappa \ge 1.0$.
/// - [`FilterError::NonFiniteInput`] if elements are NaN or infinite.
/// - [`FilterError::InvalidCovariance`] if the matrix norm is zero or non-positive.
/// - [`FilterError::NumericalInstability`] if $M$ is singular or rank-deficient.
pub fn matrix_condition_number<const N: usize>(m: &[[Real; N]; N]) -> Result<Real, FilterError> {
    for row in m.iter() {
        for val in row.iter() {
            if !val.is_finite() {
                return Err(FilterError::NonFiniteInput);
            }
        }
    }

    let norm_m = matrix_norm_1(m);
    if norm_m <= 0.0 {
        return Err(FilterError::InvalidCovariance);
    }

    let inv = invert_matrix(m)?;
    let norm_inv = matrix_norm_1(&inv);

    let cond = norm_m * norm_inv;
    if !cond.is_finite() || cond < 1.0 {
        return Ok(1.0);
    }

    Ok(cond)
}

/// Validates whether an $N \times N$ covariance matrix is well-conditioned and physically admissible.
///
/// Verification rules:
/// 1. `max_condition_number` must be finite and strictly positive.
/// 2. All elements must be finite numbers.
/// 3. All main diagonal variances must be strictly positive ($M_{ii} > 0$).
/// 4. Matrix must be approximately symmetric within numerical tolerance.
/// 5. Condition number $\kappa_1(M) \le \text{max\_condition\_number}$.
///
/// Returns `Ok(cond)` on success, or [`FilterError::InvalidCovariance`] if ill-conditioned or asymmetric.
#[allow(clippy::needless_range_loop)]
pub fn check_matrix_conditioning<const N: usize>(
    m: &[[Real; N]; N],
    max_condition_number: Real,
) -> Result<Real, FilterError> {
    if !max_condition_number.is_finite() || max_condition_number <= 0.0 {
        return Err(FilterError::InvalidParameter);
    }

    // Check diagonal variances
    for i in 0..N {
        if m[i][i] <= 0.0 || !m[i][i].is_finite() {
            return Err(FilterError::InvalidCovariance);
        }
    }

    // Check approximate symmetry
    let norm = matrix_norm_1(m);
    #[cfg(not(feature = "single-precision"))]
    let sym_tol = norm * 1e-12 + 1e-15;
    #[cfg(feature = "single-precision")]
    let sym_tol = norm * 1e-5 + 1e-6;

    for i in 0..N {
        for j in (i + 1)..N {
            if (m[i][j] - m[j][i]).abs_val() > sym_tol {
                return Err(FilterError::InvalidCovariance);
            }
        }
    }

    let cond = matrix_condition_number(m)?;
    if cond > max_condition_number {
        return Err(FilterError::InvalidCovariance);
    }

    Ok(cond)
}
