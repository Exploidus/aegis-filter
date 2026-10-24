//! Classical Linear & Kinematic Kalman Filters with Joseph stabilized covariance.

use crate::common::{FilterError, FloatOps};

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
    x: f64,
    p: f64,
    q: f64,
    r: f64,
}

impl ScalarKalmanFilter {
    pub fn new(
        initial_state: f64,
        initial_variance: f64,
        process_noise: f64,
        measurement_noise: f64,
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
        })
    }

    pub fn predict(&mut self, control_input: f64) -> Result<(), FilterError> {
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

    pub fn update(
        &mut self,
        measurement: f64,
        gate_threshold_sigma: Option<f64>,
    ) -> Result<f64, FilterError> {
        if !measurement.is_finite() {
            return Err(FilterError::NonFiniteInput);
        }

        let y = measurement - self.x;
        let s = self.p + self.r;
        if s <= 1e-15 || !s.is_finite() {
            return Err(FilterError::SingularInnovation);
        }

        if let Some(gate_sigma) = gate_threshold_sigma {
            if gate_sigma > 0.0 {
                let normalized_residual = y.abs_val() / s.sqrt_val();
                if normalized_residual > gate_sigma {
                    return Err(FilterError::MeasurementGated);
                }
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
    pub fn state(&self) -> f64 {
        self.x
    }

    #[inline(always)]
    pub fn variance(&self) -> f64 {
        self.p
    }
}

/// Generic N-dimensional Linear Kalman Filter using const-generics.
#[derive(Debug, Clone)]
pub struct LinearKalmanFilter<const S: usize, const M: usize> {
    pub x: [f64; S],
    pub p: [[f64; S]; S],
    pub f: [[f64; S]; S],
    pub q: [[f64; S]; S],
    pub h: [[f64; S]; M],
    pub r: [[f64; M]; M],
}

impl<const S: usize, const M: usize> LinearKalmanFilter<S, M> {
    pub fn new(
        initial_x: [f64; S],
        initial_p: [[f64; S]; S],
        f: [[f64; S]; S],
        q: [[f64; S]; S],
        h: [[f64; S]; M],
        r: [[f64; M]; M],
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

        Ok(Self {
            x: initial_x,
            p: initial_p,
            f,
            q,
            h,
            r,
        })
    }

    fn validate_matrix<const R: usize, const C: usize>(
        matrix: &[[f64; C]; R],
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

        self.x = x_pred;
        self.p = p_pred;
        Ok(())
    }

    /// Fuse a length-`M` measurement `z` into the state estimate.
    ///
    /// This generalises the 1D [`ScalarKalmanFilter::update`] to the multi-dimensional
    /// case, keeping the same guarantees:
    ///
    /// * **Joseph stabilised covariance update.** After `P = (I - K H) P (I - K H)^T + K S K^T`
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
    #[allow(clippy::needless_range_loop)] // manual index loops over const-generic buffers
    pub fn update(
        &mut self,
        measurement: &[f64; M],
        gate_threshold_sigma: Option<f64>,
    ) -> Result<[f64; S], FilterError> {
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
        let mut s = [[0.0; M]; M];
        for j in 0..M {
            for k in 0..M {
                let mut acc = 0.0;
                for i in 0..S {
                    // (H P H^T)[j][k] = sum_i (sum_l H[j][l] P[l][i]) H[k][i]
                    let mut hp_ji = 0.0;
                    for l in 0..S {
                        hp_ji += h[j][l] * p[l][i];
                    }
                    acc += hp_ji * h[k][i];
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
        if let Some(gate_sigma) = gate_threshold_sigma {
            if gate_sigma > 0.0 {
                let mut stat = 0.0;
                for j in 0..M {
                    for k in 0..M {
                        stat += y[j] * s_inv[j][k] * y[k];
                    }
                }
                // stat is theoretically >= 0; guard round-off that could dip below zero.
                let mahal = stat.sqrt_val();
                if mahal > gate_sigma {
                    return Err(FilterError::MeasurementGated);
                }
            }
        }

        // Kalman gain K = P H^T S^-1  (S x M).
        let mut k = [[0.0; M]; S];
        for i in 0..S {
            for j in 0..M {
                // (P H^T)[i][l] = sum_t P[i][t] H[j][t]
                let mut phi_il = 0.0;
                for t in 0..S {
                    phi_il += p[i][t] * h[j][t];
                }
                let mut acc = 0.0;
                for l in 0..M {
                    acc += phi_il * s_inv[l][j];
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

        // Joseph covariance update: P = (I - K H) P (I - K H)^T + K S K^T.
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

        let mut p_new = [[0.0; S]; S];
        for i in 0..S {
            for j in 0..S {
                // (I - K H) P (I - K H)^T = sum_{a,b} e[i][a] p[a][b] e[j][b]
                let mut ep_e = 0.0;
                for a in 0..S {
                    for b in 0..S {
                        ep_e += e[i][a] * p[a][b] * e[j][b];
                    }
                }
                // K S K^T = sum_t (sum_l k[i][l] s[l][t]) k[j][t]
                let mut k_s_k = 0.0;
                for t in 0..M {
                    let mut ks = 0.0;
                    for l in 0..M {
                        ks += k[i][l] * s[l][t];
                    }
                    k_s_k += ks * k[j][t];
                }
                p_new[i][j] = ep_e + k_s_k;
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
    fn invert_matrix<const N: usize>(m: &[[f64; N]; N]) -> Result<[[f64; N]; N], FilterError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    // (1) predict regression: a constant-velocity model must still propagate x.
    // F = [[1, 0.1], [0, 1]] -> position advances by 0.1*velocity per step; with
    // velocity held at 100, x[0] grows by 10.0 each predict. Deterministic.
    #[test]
    fn test_linear_predict_kinematic() {
        let mut f = LinearKalmanFilter::<2, 1>::new(
            [0.0, 100.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.1], [0.0, 1.0]],
            [[0.001, 0.0], [0.0, 0.001]],
            [[1.0, 0.0]], // H: one row, position only
            [[0.5]],
        )
        .unwrap();
        for _ in 0..3 {
            f.predict().unwrap();
        }
        assert!((f.x[0] - 30.0).abs() < 1e-9, "x[0] = {}", f.x[0]);
    }

    // (2) M=1 measurement update converges the state to a constant target, exactly
    // like the 1D scalar filter. S=1, M=1 with F=I: x tracks the target as the
    // posterior variance shrinks.
    #[test]
    fn test_linear_update_m1_convergence() {
        let mut f =
            LinearKalmanFilter::<1, 1>::new([0.0], [[10.0]], [[1.0]], [[0.01]], [[1.0]], [[0.5]])
                .unwrap();
        let target = 25.0;
        for _ in 0..50 {
            f.predict().unwrap();
            let _ = f.update(&[target], None).unwrap();
        }
        assert!((f.x[0] - target).abs() < 0.2, "x[0] = {}", f.x[0]);
    }

    // (3) The H=I, P=I, R=I case has an exact closed-form first update: K = P H^T
    // S^-1 = I . I . (2I)^-1 = 0.5 I, so x moves halfway to the measurement in one step.
    #[test]
    fn test_linear_update_m2_single_step_halfway() {
        let mut f = LinearKalmanFilter::<2, 2>::new(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]], // F = I
            [[0.0, 0.0], [0.0, 0.0]], // Q = 0
            [[1.0, 0.0], [0.0, 1.0]], // H = I
            [[1.0, 0.0], [0.0, 1.0]], // R = I
        )
        .unwrap();
        let z = [10.0, 20.0];
        let out = f.update(&z, None).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-9, "out[0] = {}", out[0]);
        assert!((out[1] - 10.0).abs() < 1e-9, "out[1] = {}", out[1]);
    }

    // (4) Repeated M=2 updates converge to the true measurement vector.
    #[test]
    fn test_linear_update_m2_convergence() {
        let mut f = LinearKalmanFilter::<2, 2>::new(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.01, 0.0], [0.0, 0.01]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();
        let z = [10.0, 20.0];
        for _ in 0..50 {
            f.predict().unwrap();
            f.update(&z, None).unwrap();
        }
        assert!((f.x[0] - 10.0).abs() < 0.1, "x[0] = {}", f.x[0]);
        assert!((f.x[1] - 20.0).abs() < 0.1, "x[1] = {}", f.x[1]);
    }

    // (5) Joseph form must keep P symmetric and positive on the diagonals. For the
    // H=I, P=I, R=I case the posterior covariance is the analytic 0.75 * I.
    #[test]
    fn test_linear_update_joseph_symmetry_and_psd() {
        let mut f = LinearKalmanFilter::<2, 2>::new(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();
        f.update(&[5.0, 5.0], None).unwrap();
        // Symmetry: off-diagonals match on both triangles.
        assert!((f.p[0][1] - f.p[1][0]).abs() < 1e-9, "off-diag mismatch");
        assert!((f.p[0][1]).abs() < 1e-9, "off-diag = {}", f.p[0][1]);
        // Diagonals: analytic 0.75, and strictly positive (PSD lower bound).
        assert!((f.p[0][0] - 0.75).abs() < 1e-9, "p00 = {}", f.p[0][0]);
        assert!((f.p[1][1] - 0.75).abs() < 1e-9, "p11 = {}", f.p[1][1]);
    }

    // (6) Gating: a far-off M=1 measurement is rejected (3-sigma gate) and the state
    // is left untouched, mirroring the scalar filter's MeasurementGated behaviour.
    #[test]
    fn test_linear_update_m1_gating_rejects_outlier() {
        let mut f =
            LinearKalmanFilter::<1, 1>::new([0.0], [[1.0]], [[1.0]], [[0.0]], [[1.0]], [[0.01]])
                .unwrap();
        let err = f.update(&[1000.0], Some(3.0));
        assert_eq!(err, Err(FilterError::MeasurementGated));
        assert!((f.x[0] - 0.0).abs() < 1e-12, "state moved on gated update");
    }

    // (7) A rank-deficient observation (both rows identical) with zero process and
    // measurement noise makes the innovation covariance singular -> NumericalInstability.
    #[test]
    fn test_linear_update_singular_innovation() {
        let mut f = LinearKalmanFilter::<2, 2>::new(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]], // rank-deficient H
            [[0.0, 0.0], [0.0, 0.0]], // R = 0, so S = H P H^T is singular
        )
        .unwrap();
        assert_eq!(
            f.update(&[1.0, 1.0], None).err(),
            Some(FilterError::NumericalInstability)
        );
    }

    // (8) Non-finite measurements are rejected at the boundary, like the 1D path.
    #[test]
    fn test_linear_update_nonfinite_measurement() {
        let mut f =
            LinearKalmanFilter::<1, 1>::new([0.0], [[1.0]], [[1.0]], [[0.01]], [[1.0]], [[0.5]])
                .unwrap();
        assert_eq!(
            f.update(&[f64::NAN], None).err(),
            Some(FilterError::NonFiniteInput)
        );
        assert_eq!(
            f.update(&[f64::INFINITY], None).err(),
            Some(FilterError::NonFiniteInput)
        );
    }
}
