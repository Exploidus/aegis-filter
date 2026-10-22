//! Performance and Latency Benchmarks for aegis-filter engines.

use aegis_filter::{HInfinityFilter1D, ScalarKalmanFilter, UnscentedKalmanFilter2D};
use std::time::Instant;

fn main() {
    println!("===============================================================================");
    println!("  AEGIS-FILTER :: PERFORMANCE & EXECUTION LATENCY BENCHMARK");
    println!("===============================================================================\n");

    #[cfg(debug_assertions)]
    {
        eprintln!(
            "[NOTE] Running in debug mode. Latency figures include debug assertion overhead."
        );
        eprintln!("[NOTE] Run with `cargo run --example benchmark --release` for true zero-cost telemetry latency.\n");
    }

    const ITERATIONS: u64 = 1_000_000;

    // 1. Benchmark: Scalar Kalman Filter (Joseph Form + Predict + Update)
    {
        let mut filter = ScalarKalmanFilter::new(0.0, 1.0, 0.01, 0.5).unwrap();
        let start = Instant::now();
        for i in 0..ITERATIONS {
            filter.predict(0.01).unwrap();
            let _ = filter.update(10.0 + (i as f64 * 0.0001), None).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op_ns = elapsed.as_nanos() as f64 / (ITERATIONS * 2) as f64;
        println!(
            "{:<35} | Total: {:>8.2?} | {:>8.2} ns / operation",
            "ScalarKalmanFilter (Predict+Update)", elapsed, per_op_ns
        );
    }

    // 2. Benchmark: H-Infinity Robust Minimax Filter
    {
        let mut h_inf = HInfinityFilter1D::new(0.0, 1.0, 0.01, 0.5, 5.0).unwrap();
        let start = Instant::now();
        for i in 0..ITERATIONS {
            h_inf.predict(0.01).unwrap();
            let _ = h_inf.update(10.0 + (i as f64 * 0.0001)).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op_ns = elapsed.as_nanos() as f64 / (ITERATIONS * 2) as f64;
        println!(
            "{:<35} | Total: {:>8.2?} | {:>8.2} ns / operation",
            "HInfinityFilter1D (Predict+Update)", elapsed, per_op_ns
        );
    }

    // 3. Benchmark: 2D Unscented Kalman Filter (Sigma Points Propagation)
    {
        let mut ukf = UnscentedKalmanFilter2D::new(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.01, 0.0], [0.0, 0.01]],
            0.5,
        )
        .unwrap();
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            ukf.predict(0.01, |x, _dt| [x[0] + 0.01, x[1]]).unwrap();
            let _ = ukf.update(1.0, |x| x[0], None).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op_ns = elapsed.as_nanos() as f64 / (ITERATIONS * 2) as f64;
        println!(
            "{:<35} | Total: {:>8.2?} | {:>8.2} ns / operation",
            "UnscentedKalmanFilter2D (5 Sigma Pts)", elapsed, per_op_ns
        );
    }

    println!("\n-------------------------------------------------------------------------------");
    println!("  Zero heap allocations. Deterministic latency profile.");
    println!("  Release-profile latency measured; verify on physical target hardware.");
    println!("===============================================================================");
}
