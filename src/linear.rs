//! Classical Linear & Kinematic Kalman Filters with Joseph stabilized covariance.

use crate::common::{FilterError, FloatOps};

/// 1D Scalar Kalman Filter with Joseph-stabilized variance update.
#[derive(Debug, Clone, Copy)]
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
}
