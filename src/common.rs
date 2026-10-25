//! Core mathematical types, errors, and `#![no_std]` floating point helpers.

/// Standard recommended 3-sigma innovation threshold for outlier and anomaly rejection.
pub const DEFAULT_OUTLIER_GATE_SIGMA: f64 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FilterError {
    /// Provided float was NaN or infinite.
    NonFiniteInput,
    /// Provided covariance is non-positive definite or ill-conditioned.
    InvalidCovariance,
    /// Innovation covariance is singular or ill-conditioned for inversion.
    SingularInnovation,
    /// Measurement exceeded gating threshold and was rejected as an outlier.
    MeasurementGated,
    /// Internal matrix inversion failure or non-positive square root.
    NumericalInstability,
    /// Invalid parameter value provided (e.g., negative variance or zero threshold).
    InvalidParameter,
    /// Static capacity exceeded for fixed-size sensor buffers.
    CapacityExceeded,
}

impl core::fmt::Display for FilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteInput => write!(
                f,
                "Input contains non-finite floating point values (NaN or Inf)"
            ),
            Self::InvalidCovariance => write!(f, "Covariance is non-positive or ill-conditioned"),
            Self::SingularInnovation => {
                write!(f, "Innovation covariance is singular or ill-conditioned")
            }
            Self::MeasurementGated => write!(
                f,
                "Measurement was gated out as an outlier or adversarial spike"
            ),
            Self::NumericalInstability => write!(
                f,
                "Numerical instability occurred during matrix factorization or square root"
            ),
            Self::InvalidParameter => write!(f, "Invalid parameter value provided to filter"),
            Self::CapacityExceeded => write!(f, "Maximum registered sensor capacity exceeded"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FilterError {}

/// Trait providing safe, deterministic `#![no_std]` math operations for f64.
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
