# aegis-filter

High-integrity, target-deterministic (on a given ISA/FPU), and `#![no_std]` capable state estimation and robust filtering engine written in pure Rust.

Designed for mission-critical robotics, aerospace, navigation, autonomous sensor fusion, and embedded telemetry where **numerical stability**, **fail-safe error handling**, and **adversarial/outlier resilience** are paramount.

Developed under **[Exploidus](https://github.com/Exploidus)**.

---

## Architectural & Security Highlights

- **Joseph Stabilized Covariance Updates**: Standard Kalman Filter updates ($\mathbf{P} = (\mathbf{I} - \mathbf{KH})\mathbf{P}$) frequently become asymmetric and lose positive semi-definiteness under finite floating-point round-off. `aegis-filter` implements the **Joseph Form** covariance update:
  $$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^- (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k \mathbf{R} \mathbf{K}_k^T$$
  which unconditionally guarantees symmetry and positive semi-definiteness.
- **Strict Non-Finite Float Guards**: Hardened against `NaN`, `+Inf`, and `-Inf` on all state vectors, inputs, and measurements. All methods return `Result<T, KalmanError>` with zero panics.
- **Mahalanobis Distance / Chi-Square Outlier Gating**: Built-in residual gating to discard wild sensor glitches, adversarial spoofed signals, or hardware spikes without contaminating the filter state.
- **Pure `#![no_std]` & Zero Heap Allocations**: Const generic dimensions (`KalmanFilter<STATES, MEASUREMENTS>`) allow running directly on bare-metal microcontrollers (ARM Cortex-M, RISC-V) without `alloc` or OS dependencies.

---

## Filter & Telemetry Architectures

1. **`ScalarKalmanFilter`**: Ultra-lightweight 1D scalar filter for temperature, pressure altitude, or single-variable telemetry tracking.
2. **`LinearKalmanFilter<STATES, MEASUREMENTS>`**: Fully generic multi-dimensional linear Kalman filter configured via compile-time const generics.
3. **`MultiSensorFusion1D<CAPACITY>`**: Multi-sensor redundant fusion engine with dynamic sensor health scoring, automated Fault Detection, Isolation, and Recovery (FDIR), and consecutive-health auto-recovery.
4. **`UnscentedKalmanFilter2D`**: Non-linear estimation via Merwe scaled sigma points without numerical Jacobian derivations.
5. **`HInfinityFilter1D`**: Minimax bounded worst-case disturbance filter for unmodeled dynamics.
6. **`AdaptiveFilter1D`**: Online covariance estimation adapting to non-stationary sensor noise.
7. **`WindowedStats<CAPACITY>`**: Zero-allocation sliding-window statistical analyzer computing rolling mean, variance, standard deviation, and peak-to-peak spread over fixed const-generic buffers.

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
aegis-filter = "0.3.0"
```

For `#![no_std]` environments (bare-metal):

```toml
[dependencies]
aegis-filter = { version = "0.3.0", default-features = false }
```

With optional `serde` support for state persistence and telemetry serialization:

```toml
[dependencies]
aegis-filter = { version = "0.3.0", features = ["serde"] }
```

---

## Example Usage

### 1D Sensor Fusion with Adversarial Outlier Gating

```rust
use aegis_filter::{ScalarKalmanFilter, FilterError};

fn main() -> Result<(), FilterError> {
    // Initial state: 20.0, Initial variance: 5.0, Process noise (Q): 0.01, Measurement noise (R): 0.5
    let mut filter = ScalarKalmanFilter::new(20.0, 5.0, 0.01, 0.5)?;

    let sensor_stream = [20.2, 20.4, 20.1, 999.0 /* spoofed spike */, 20.3];

    for &reading in &sensor_stream {
        filter.predict(0.0)?;

        // Update with 3-sigma outlier rejection threshold
        match filter.update(reading, Some(3.0)) {
            Ok(estimated_state) => {
                println!("Valid state estimate: {:.3}", estimated_state);
            }
            Err(FilterError::MeasurementGated) => {
                println!("Sensor glitch detected and safely rejected: {}", reading);
            }
            Err(err) => return Err(err),
        }
    }

    Ok(())
}
```

---

## Interactive Simulations & Benchmarks

The repository includes runnable examples showcasing real-world aerospace and robotics applications:

```bash
# 1. Flight Telemetry & Spoofing Resilience (3-sigma gating under GPS/baro spikes)
cargo run --example flight_telemetry

# 2. Non-linear Radar Tracking via UKF (aircraft range tracking without Jacobians)
cargo run --example radar_tracking_ukf

# 3. Execution Latency & Nanosecond Benchmark (1,000,000 iterations profile)
cargo run --example benchmark

# 4. Multi-Sensor Fusion & Automated FDIR (GPS spoof quarantine & Baro/LiDAR handover)
cargo run --example sensor_fault_fdir
```

---

## License

Dual-licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
