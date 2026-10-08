# kalman-filter-rs

High-integrity, deterministic, and `#![no_std]` capable Kalman Filtering engine written in pure Rust.

Designed for mission-critical robotics, aerospace, navigation, automotive sensor fusion, and embedded telemetry where **numerical stability**, **fail-safe error handling**, and **fault/outlier resilience** are paramount.

---

## Key Architectural & Security Features

- **Joseph Stabilized Covariance Updates**: Standard Kalman Filter updates (`P = (I - KH)P`) can become asymmetric and lose positive semi-definiteness under finite floating-point precision round-off. `kalman-filter-rs` implements the **Joseph Form** covariance update:
  $$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^- (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k \mathbf{R} \mathbf{K}_k^T$$
  which guarantees symmetry and positive semi-definiteness unconditionally.
- **Strict Non-Finite Float Guards**: Rejects `NaN`, `+Inf`, and `-Inf` on all state entries, control inputs, and measurements. All methods return `Result<T, KalmanError>` with zero panics.
- **Covariance Conditioning & Non-Degeneracy**: Protects against non-positive variances and division by near-zero innovation covariances.
- **Mahalanobis Distance / Chi-Square Outlier Gating**: Built-in residual gating to discard wild sensor glitches, spoofed signals, or faulty sensor spikes without contaminating the filter state.
- **Pure `#![no_std]` & Zero Heap Allocations**: Const generic dimensions (`KalmanFilter<STATES, MEASUREMENTS>`) allow running directly on bare-metal microcontrollers (ARM Cortex-M, RISC-V) without `alloc` or OS dependencies.

---

## Components

1. **`ScalarKalmanFilter`**: Ultra-lightweight 1D scalar filter for temperature, barometer, or single-variable telemetry tracking.
2. **`Kinematic1DFilter`**: 2-state / 1-measurement filter tracking position and velocity from a position sensor (e.g., GPS latitude/longitude, rangefinder) with continuous white-noise process modeling.
3. **`KalmanFilter<STATES, MEASUREMENTS>`**: Fully generic multi-dimensional linear Kalman filter configured via const generics.

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
kalman-filter-rs = "0.1.0"
```

For `#![no_std]` environments (bare-metal):

```toml
[dependencies]
kalman-filter-rs = { version = "0.1.0", default-features = false }
```

---

## Example Usage

### 1D Sensor Fusion with Outlier Gating

```rust
use kalman_filter_rs::{ScalarKalmanFilter, KalmanError};

fn main() -> Result<(), KalmanError> {
    // Initial state: 20.0, Initial variance: 5.0, Process noise (Q): 0.01, Measurement noise (R): 0.5
    let mut filter = ScalarKalmanFilter::new(20.0, 5.0, 0.01, 0.5)?;

    let sensor_stream = [20.2, 20.4, 20.1, 999.0 /* glitch */, 20.3];

    for &reading in &sensor_stream {
        filter.predict(0.0)?;

        // Update with 3-sigma outlier rejection threshold
        match filter.update(reading, Some(3.0)) {
            Ok(estimated_state) => {
                println!("Valid state estimate: {:.3}", estimated_state);
            }
            Err(KalmanError::MeasurementGated) => {
                println!("Sensor glitch detected and safely rejected: {}", reading);
            }
            Err(err) => return Err(err),
        }
    }

    Ok(())
}
```

---

## License

Dual-licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
