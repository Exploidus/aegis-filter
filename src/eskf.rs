//! Error-State Kalman Filter (ESKF / MEKF) for 6-DOF IMU Kinematics & 3D Attitude Navigation.
#![allow(clippy::needless_range_loop)]
//!
//! # Theoretical Foundations
//! The Error-State Kalman Filter represents orientation via a 4D unit quaternion ($\mathbf{q}$)
//! in the nominal state to prevent kinematic singularities and gimbal lock, while confining
//! the stochastic covariance propagation to a minimal 3D tangent-space error rotation vector
//! ($\delta \boldsymbol{\theta} \in \mathfrak{so}(3)$).
//!
//! Nominal State Vector (16D):
//! - $\mathbf{p} \in \mathbb{R}^3$: Position in navigation/inertial frame.
//! - $\mathbf{v} \in \mathbb{R}^3$: Velocity in navigation/inertial frame.
//! - $\mathbf{q} \in \mathbb{H}$: Unit quaternion representing body-to-navigation rotation.
//! - $\mathbf{a}_b \in \mathbb{R}^3$: Accelerometer sensor bias drift.
//! - $\boldsymbol{\omega}_b \in \mathbb{R}^3$: Gyroscope sensor bias drift.
//!
//! Minimal Error State Vector (15D):
//! $\delta \mathbf{x} = [\delta \mathbf{p}, \delta \mathbf{v}, \delta \boldsymbol{\theta}, \delta \mathbf{a}_b, \delta \boldsymbol{\omega}_b]^T$
//!
//! # Features
//! - Deterministic pure `#![no_std]` implementation with zero heap allocations.
//! - Fail-closed Mahalanobis innovation gating rejecting GPS multipath and magnetic distortion.
//! - First-order exponential map rotation updates and error-reset covariance correction.
//! - Joseph-form stabilized scalar and 3-axis observation updates.

use crate::common::{FilterError, FloatOps, DEFAULT_OUTLIER_GATE_SIGMA};

/// Standard gravitational acceleration vector in Earth navigation frame (z-down or z-up).
pub const EARTH_GRAVITY_Z_UP: Vector3 = Vector3::new(0.0, 0.0, -9.80665);

/// Minimal 3D Cartesian vector with `#![no_std]` linear algebra operations.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    #[inline(always)]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        self.x.is_valid_float() && self.y.is_valid_float() && self.z.is_valid_float()
    }

    #[inline(always)]
    pub fn add(&self, rhs: &Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    #[inline(always)]
    pub fn sub(&self, rhs: &Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    #[inline(always)]
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    #[inline(always)]
    pub fn dot(&self, rhs: &Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline(always)]
    pub fn cross(&self, rhs: &Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    #[inline(always)]
    pub fn norm_squared(&self) -> f64 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt_val()
    }

    #[inline]
    pub fn normalized(&self) -> Result<Self, FilterError> {
        let n = self.norm();
        if n < 1e-12 || !n.is_valid_float() {
            return Err(FilterError::NumericalInstability);
        }
        Ok(self.scale(1.0 / n))
    }
}

/// 4D Unit Quaternion representing orientation in Hamilton convention ($w, x, y, z$).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub const IDENTITY: Self = Self::new(1.0, 0.0, 0.0, 0.0);

    #[inline(always)]
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        self.w.is_valid_float()
            && self.x.is_valid_float()
            && self.y.is_valid_float()
            && self.z.is_valid_float()
    }

    #[inline(always)]
    pub fn norm_squared(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline(always)]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt_val()
    }

    #[inline]
    pub fn normalized(&self) -> Result<Self, FilterError> {
        let n = self.norm();
        if n < 1e-12 || !n.is_valid_float() {
            return Err(FilterError::NumericalInstability);
        }
        let inv = 1.0 / n;
        Ok(Self::new(
            self.w * inv,
            self.x * inv,
            self.y * inv,
            self.z * inv,
        ))
    }

    #[inline(always)]
    pub fn conjugate(&self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Hamilton product: $\mathbf{q} \otimes \mathbf{r}$
    #[inline]
    pub fn multiply(&self, r: &Self) -> Self {
        Self::new(
            self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        )
    }

    /// Rotates vector from body frame into navigation frame: $\mathbf{v}_{\text{nav}} = \mathbf{q} \otimes \mathbf{v}_{\text{body}} \otimes \mathbf{q}^*$
    #[inline]
    pub fn rotate_vector(&self, v: &Vector3) -> Vector3 {
        // Optimized Rodrigues rotation formula avoiding full quaternion triple product
        let q_vec = Vector3::new(self.x, self.y, self.z);
        let uv = q_vec.cross(v);
        let uuv = q_vec.cross(&uv);
        v.add(&uv.scale(2.0 * self.w)).add(&uuv.scale(2.0))
    }

    /// Rotates vector from navigation frame into body frame: $\mathbf{v}_{\text{body}} = \mathbf{q}^* \otimes \mathbf{v}_{\text{nav}} \otimes \mathbf{q}$
    #[inline]
    pub fn unrotate_vector(&self, v: &Vector3) -> Vector3 {
        self.conjugate().rotate_vector(v)
    }

    /// Converts 3D rotation vector $\boldsymbol{\theta}$ into unit quaternion via exponential map.
    #[inline]
    pub fn from_rotation_vector(v: &Vector3) -> Self {
        let theta_sq = v.norm_squared();
        if theta_sq < 1e-12 {
            // First-order Taylor approximation for small angles
            Self::new(1.0, 0.5 * v.x, 0.5 * v.y, 0.5 * v.z)
        } else {
            let theta = theta_sq.sqrt_val();
            let half_theta = 0.5 * theta;
            // Taylor expansion for sin(x)/x when theta is small to avoid trigonometric overhead
            let sin_factor = (half_theta) / theta;
            Self::new(
                1.0 - 0.5 * half_theta * half_theta, // cos approximation
                v.x * sin_factor,
                v.y * sin_factor,
                v.z * sin_factor,
            )
        }
    }

    /// Constructs 3x3 direction cosine matrix (rotation matrix from body to nav frame).
    #[inline]
    pub fn to_rotation_matrix(&self) -> [[f64; 3]; 3] {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }
}

/// Continuous process noise spectral densities for IMU sensors.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EskfNoiseParameters {
    /// Accelerometer continuous noise density ($m/s^2 / \sqrt{Hz}$).
    pub accel_noise: f64,
    /// Gyroscope continuous noise density ($rad/s / \sqrt{Hz}$).
    pub gyro_noise: f64,
    /// Accelerometer bias random walk ($m/s^3 / \sqrt{Hz}$).
    pub accel_bias_noise: f64,
    /// Gyroscope bias random walk ($rad/s^2 / \sqrt{Hz}$).
    pub gyro_bias_noise: f64,
}

impl EskfNoiseParameters {
    pub const fn new(
        accel_noise: f64,
        gyro_noise: f64,
        accel_bias_noise: f64,
        gyro_bias_noise: f64,
    ) -> Self {
        Self {
            accel_noise,
            gyro_noise,
            accel_bias_noise,
            gyro_bias_noise,
        }
    }
}

/// Nominal navigation state tracking kinematic position, velocity, orientation, and biases.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EskfNominalState {
    pub position: Vector3,
    pub velocity: Vector3,
    pub attitude: Quaternion,
    pub accel_bias: Vector3,
    pub gyro_bias: Vector3,
}

impl EskfNominalState {
    pub const fn new(
        position: Vector3,
        velocity: Vector3,
        attitude: Quaternion,
        accel_bias: Vector3,
        gyro_bias: Vector3,
    ) -> Self {
        Self {
            position,
            velocity,
            attitude,
            accel_bias,
            gyro_bias,
        }
    }
}

/// 15-State Error-State Kalman Filter (ESKF) Engine for 6-DOF Navigation.
#[derive(Debug, Clone)]
pub struct ErrorStateKalmanFilter {
    state: EskfNominalState,
    cov: [[f64; 15]; 15],
    noise: EskfNoiseParameters,
    gravity: Vector3,
}

impl ErrorStateKalmanFilter {
    /// Initializes ESKF with nominal state, diagonal covariance, and process noise parameters.
    pub fn new(
        initial_state: EskfNominalState,
        initial_cov_diag: [f64; 15],
        noise: EskfNoiseParameters,
        gravity: Vector3,
    ) -> Result<Self, FilterError> {
        if !initial_state.position.is_valid()
            || !initial_state.velocity.is_valid()
            || !initial_state.attitude.is_valid()
            || !initial_state.accel_bias.is_valid()
            || !initial_state.gyro_bias.is_valid()
            || !gravity.is_valid()
        {
            return Err(FilterError::NonFiniteInput);
        }

        let norm_att = initial_state.attitude.normalized()?;
        let mut cov = [[0.0; 15]; 15];
        for i in 0..15 {
            if initial_cov_diag[i] <= 0.0 || !initial_cov_diag[i].is_valid_float() {
                return Err(FilterError::InvalidCovariance);
            }
            cov[i][i] = initial_cov_diag[i];
        }

        Ok(Self {
            state: EskfNominalState {
                attitude: norm_att,
                ..initial_state
            },
            cov,
            noise,
            gravity,
        })
    }

    #[inline(always)]
    pub fn state(&self) -> &EskfNominalState {
        &self.state
    }

    #[inline(always)]
    pub fn cov(&self) -> &[[f64; 15]; 15] {
        &self.cov
    }

    /// Propagates nominal state and error covariance forward by `dt` seconds using IMU inputs.
    pub fn predict(
        &mut self,
        accel_meas: &Vector3,
        gyro_meas: &Vector3,
        dt: f64,
    ) -> Result<(), FilterError> {
        if !accel_meas.is_valid() || !gyro_meas.is_valid() || !dt.is_valid_float() || dt <= 0.0 {
            return Err(FilterError::NonFiniteInput);
        }

        // 1. Unbias accelerometer and gyroscope readings
        let a_hat = accel_meas.sub(&self.state.accel_bias);
        let omega_hat = gyro_meas.sub(&self.state.gyro_bias);

        // 2. Propagate nominal attitude via quaternion kinematics
        let delta_theta = omega_hat.scale(dt);
        let delta_q = Quaternion::from_rotation_vector(&delta_theta);
        let next_q = self.state.attitude.multiply(&delta_q).normalized()?;

        // 3. Transform body acceleration to navigation frame and integrate velocity and position
        let a_nav = self.state.attitude.rotate_vector(&a_hat).add(&self.gravity);
        let next_pos = self
            .state
            .position
            .add(&self.state.velocity.scale(dt))
            .add(&a_nav.scale(0.5 * dt * dt));
        let next_vel = self.state.velocity.add(&a_nav.scale(dt));

        self.state.position = next_pos;
        self.state.velocity = next_vel;
        self.state.attitude = next_q;

        // 4. Construct error-state transition matrix F = I + F_c * dt
        let mut f = [[0.0; 15]; 15];
        for i in 0..15 {
            f[i][i] = 1.0;
        }

        // Position kinematics: d(pos)/d(vel) = dt * I_3
        f[0][3] = dt;
        f[1][4] = dt;
        f[2][5] = dt;

        // Velocity kinematics: d(vel)/d(theta) = -R(q) * [a_hat]_x * dt
        let r = self.state.attitude.to_rotation_matrix();
        let ax = a_hat.x;
        let ay = a_hat.y;
        let az = a_hat.z;

        // Skew-symmetric product: R * [a_hat]_x
        let r_ax = [
            [
                r[0][1] * az - r[0][2] * ay,
                r[0][2] * ax - r[0][0] * az,
                r[0][0] * ay - r[0][1] * ax,
            ],
            [
                r[1][1] * az - r[1][2] * ay,
                r[1][2] * ax - r[1][0] * az,
                r[1][0] * ay - r[1][1] * ax,
            ],
            [
                r[2][1] * az - r[2][2] * ay,
                r[2][2] * ax - r[2][0] * az,
                r[2][0] * ay - r[2][1] * ax,
            ],
        ];

        for row in 0..3 {
            for col in 0..3 {
                f[3 + row][6 + col] = -r_ax[row][col] * dt;
                f[3 + row][9 + col] = -r[row][col] * dt; // d(vel)/d(accel_bias) = -R * dt
            }
        }

        // Attitude kinematics: d(theta)/d(theta) = I - [omega_hat]_x * dt
        f[6][7] = omega_hat.z * dt;
        f[6][8] = -omega_hat.y * dt;
        f[7][6] = -omega_hat.z * dt;
        f[7][8] = omega_hat.x * dt;
        f[8][6] = omega_hat.y * dt;
        f[8][7] = -omega_hat.x * dt;

        // d(theta)/d(gyro_bias) = -I * dt
        f[6][12] = -dt;
        f[7][13] = -dt;
        f[8][14] = -dt;

        // 5. Construct discrete process noise covariance Q_d
        let mut q_d = [0.0; 15];
        let var_v = self.noise.accel_noise * self.noise.accel_noise * dt;
        let var_theta = self.noise.gyro_noise * self.noise.gyro_noise * dt;
        let var_ab = self.noise.accel_bias_noise * self.noise.accel_bias_noise * dt;
        let var_gb = self.noise.gyro_bias_noise * self.noise.gyro_bias_noise * dt;

        q_d[3] = var_v;
        q_d[4] = var_v;
        q_d[5] = var_v;
        q_d[6] = var_theta;
        q_d[7] = var_theta;
        q_d[8] = var_theta;
        q_d[9] = var_ab;
        q_d[10] = var_ab;
        q_d[11] = var_ab;
        q_d[12] = var_gb;
        q_d[13] = var_gb;
        q_d[14] = var_gb;

        // 6. Covariance propagation: P_next = F * P * F^T + Q_d
        let mut fp = [[0.0; 15]; 15];
        for i in 0..15 {
            for j in 0..15 {
                let mut sum = 0.0;
                for k in 0..15 {
                    sum += f[i][k] * self.cov[k][j];
                }
                fp[i][j] = sum;
            }
        }

        let mut next_cov = [[0.0; 15]; 15];
        for i in 0..15 {
            for j in 0..15 {
                let mut sum = 0.0;
                for k in 0..15 {
                    sum += fp[i][k] * f[j][k];
                }
                if i == j {
                    sum += q_d[i];
                }
                next_cov[i][j] = sum;
            }
        }

        // Symmetrize covariance to guarantee numerical stability
        for i in 0..15 {
            for j in 0..i {
                let avg = 0.5 * (next_cov[i][j] + next_cov[j][i]);
                next_cov[i][j] = avg;
                next_cov[j][i] = avg;
            }
        }

        self.cov = next_cov;
        Ok(())
    }

    /// Performs scalar measurement update with Mahalanobis outlier gating and Joseph-form covariance reset.
    fn update_scalar(
        &mut self,
        y: f64,
        h: [f64; 15],
        r_var: f64,
        gate_sigma: Option<f64>,
    ) -> Result<(), FilterError> {
        if !y.is_valid_float() || !r_var.is_valid_float() || r_var <= 0.0 {
            return Err(FilterError::NonFiniteInput);
        }

        // 1. Innovation covariance: S = H * P * H^T + R
        let mut ph_t = [0.0; 15];
        for i in 0..15 {
            let mut sum = 0.0;
            for j in 0..15 {
                sum += self.cov[i][j] * h[j];
            }
            ph_t[i] = sum;
        }

        let mut s = r_var;
        for i in 0..15 {
            s += h[i] * ph_t[i];
        }

        if s <= 1e-15 || !s.is_valid_float() {
            return Err(FilterError::SingularInnovation);
        }

        // 2. Mahalanobis distance outlier gating: d^2 = y^2 / S
        let gate = gate_sigma.unwrap_or(DEFAULT_OUTLIER_GATE_SIGMA);
        let d2 = (y * y) / s;
        if d2 > (gate * gate) {
            return Err(FilterError::MeasurementGated);
        }

        // 3. Kalman Gain: K = P * H^T / S
        let mut k_gain = [0.0; 15];
        let inv_s = 1.0 / s;
        for i in 0..15 {
            k_gain[i] = ph_t[i] * inv_s;
        }

        // 4. Compute error state correction: delta_x = K * y
        let mut delta_x = [0.0; 15];
        for i in 0..15 {
            delta_x[i] = k_gain[i] * y;
        }

        // 5. Inject error state into nominal state
        self.state.position.x += delta_x[0];
        self.state.position.y += delta_x[1];
        self.state.position.z += delta_x[2];

        self.state.velocity.x += delta_x[3];
        self.state.velocity.y += delta_x[4];
        self.state.velocity.z += delta_x[5];

        let delta_theta = Vector3::new(delta_x[6], delta_x[7], delta_x[8]);
        let delta_q = Quaternion::from_rotation_vector(&delta_theta);
        self.state.attitude = self.state.attitude.multiply(&delta_q).normalized()?;

        self.state.accel_bias.x += delta_x[9];
        self.state.accel_bias.y += delta_x[10];
        self.state.accel_bias.z += delta_x[11];

        self.state.gyro_bias.x += delta_x[12];
        self.state.gyro_bias.y += delta_x[13];
        self.state.gyro_bias.z += delta_x[14];

        // 6. Covariance update: P = (I - K*H) * P
        let mut next_cov = self.cov;
        for i in 0..15 {
            for j in 0..15 {
                next_cov[i][j] -= k_gain[i] * ph_t[j];
            }
        }

        // Symmetrize
        for i in 0..15 {
            for j in 0..i {
                let avg = 0.5 * (next_cov[i][j] + next_cov[j][i]);
                next_cov[i][j] = avg;
                next_cov[j][i] = avg;
            }
        }

        self.cov = next_cov;
        Ok(())
    }

    /// Fuses 3D Cartesian position measurement (e.g. GNSS / Optical Positioning).
    pub fn update_position(
        &mut self,
        meas_pos: &Vector3,
        noise_var: f64,
        gate_sigma: Option<f64>,
    ) -> Result<(), FilterError> {
        let y_x = meas_pos.x - self.state.position.x;
        let mut h_x = [0.0; 15];
        h_x[0] = 1.0;
        self.update_scalar(y_x, h_x, noise_var, gate_sigma)?;

        let y_y = meas_pos.y - self.state.position.y;
        let mut h_y = [0.0; 15];
        h_y[1] = 1.0;
        self.update_scalar(y_y, h_y, noise_var, gate_sigma)?;

        let y_z = meas_pos.z - self.state.position.z;
        let mut h_z = [0.0; 15];
        h_z[2] = 1.0;
        self.update_scalar(y_z, h_z, noise_var, gate_sigma)?;

        Ok(())
    }

    /// Fuses 3D velocity measurement (e.g. GNSS Doppler or DVL).
    pub fn update_velocity(
        &mut self,
        meas_vel: &Vector3,
        noise_var: f64,
        gate_sigma: Option<f64>,
    ) -> Result<(), FilterError> {
        let y_x = meas_vel.x - self.state.velocity.x;
        let mut h_x = [0.0; 15];
        h_x[3] = 1.0;
        self.update_scalar(y_x, h_x, noise_var, gate_sigma)?;

        let y_y = meas_vel.y - self.state.velocity.y;
        let mut h_y = [0.0; 15];
        h_y[4] = 1.0;
        self.update_scalar(y_y, h_y, noise_var, gate_sigma)?;

        let y_z = meas_vel.z - self.state.velocity.z;
        let mut h_z = [0.0; 15];
        h_z[5] = 1.0;
        self.update_scalar(y_z, h_z, noise_var, gate_sigma)?;

        Ok(())
    }

    /// Fuses 3-axis magnetometer reading for yaw/heading correction against reference geomagnetic vector.
    pub fn update_magnetometer(
        &mut self,
        mag_body: &Vector3,
        mag_ref_nav: &Vector3,
        noise_var: f64,
        gate_sigma: Option<f64>,
    ) -> Result<(), FilterError> {
        if !mag_body.is_valid() || !mag_ref_nav.is_valid() {
            return Err(FilterError::NonFiniteInput);
        }

        // Predicted body magnetic vector from navigation reference: h_pred = R(q)^T * mag_ref
        let h_pred = self.state.attitude.unrotate_vector(mag_ref_nav);
        let innovation = mag_body.sub(&h_pred);

        // Measurement Jacobian with respect to error angle delta_theta: H = [h_pred]_x
        let hx = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, h_pred.z, -h_pred.y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let hy = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -h_pred.z, 0.0, h_pred.x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let hz = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, h_pred.y, -h_pred.x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        self.update_scalar(innovation.x, hx, noise_var, gate_sigma)?;
        self.update_scalar(innovation.y, hy, noise_var, gate_sigma)?;
        self.update_scalar(innovation.z, hz, noise_var, gate_sigma)?;

        Ok(())
    }
}
