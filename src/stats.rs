//! High-integrity running and sliding-window statistics for sensor telemetry and residual monitoring.
//!
//! Designed for `#![no_std]` environments with const-generic fixed buffers and zero heap allocations.

use crate::common::{FilterError, FloatOps};

/// Deterministic, zero-allocation rolling window statistics over a fixed capacity `N`.
#[derive(Debug, Clone, Copy)]
pub struct WindowedStats<const N: usize> {
    buffer: [f64; N],
    count: usize,
    head: usize,
}

impl<const N: usize> WindowedStats<N> {
    /// Creates a new empty sliding window of capacity `N`.
    pub const fn new() -> Self {
        Self {
            buffer: [0.0; N],
            count: 0,
            head: 0,
        }
    }

    /// Resets all accumulated samples.
    pub fn clear(&mut self) {
        self.count = 0;
        self.head = 0;
    }

    /// Number of samples currently in the window (min(inserted, N)).
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the window contains no samples.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Maximum capacity of the sliding window.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Pushes a new sample into the window, overwriting the oldest sample if capacity is reached.
    pub fn push(&mut self, sample: f64) -> Result<(), FilterError> {
        if !sample.is_valid_float() {
            return Err(FilterError::NonFiniteInput);
        }
        self.buffer[self.head] = sample;
        self.head = (self.head + 1) % N;
        if self.count < N {
            self.count += 1;
        }
        Ok(())
    }

    /// Computes the arithmetic mean of the samples in the current window.
    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let mut sum = 0.0;
        for i in 0..self.count {
            sum += self.buffer[i];
        }
        Some(sum / self.count as f64)
    }

    /// Computes sample variance of the windowed samples.
    pub fn variance(&self) -> Option<f64> {
        if self.count < 2 {
            return Some(0.0);
        }
        let m = self.mean()?;
        let mut sum_sq = 0.0;
        for i in 0..self.count {
            let diff = self.buffer[i] - m;
            sum_sq += diff * diff;
        }
        Some(sum_sq / (self.count - 1) as f64)
    }

    /// Computes standard deviation of the windowed samples.
    pub fn std_dev(&self) -> Option<f64> {
        let v = self.variance()?;
        Some(v.sqrt_val())
    }

    /// Returns the minimum value in the current window.
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let mut min_val = self.buffer[0];
        for i in 1..self.count {
            if self.buffer[i] < min_val {
                min_val = self.buffer[i];
            }
        }
        Some(min_val)
    }

    /// Returns the maximum value in the current window.
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let mut max_val = self.buffer[0];
        for i in 1..self.count {
            if self.buffer[i] > max_val {
                max_val = self.buffer[i];
            }
        }
        Some(max_val)
    }

    /// Returns peak-to-peak spread (max - min).
    pub fn peak_to_peak(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        Some(self.max()? - self.min()?)
    }
}

impl<const N: usize> Default for WindowedStats<N> {
    fn default() -> Self {
        Self::new()
    }
}
