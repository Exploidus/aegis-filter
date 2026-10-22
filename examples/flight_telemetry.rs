//! Flight Telemetry & Spoofing Resilience Simulation.
//!
//! Demonstrates:
//! 1. Flight Altitude Tracking under Noisy Atmospheric Barometer.
//! 2. Adversarial Sensor Spoofing / Malicious Spikes Rejected by Outlier Gating.
//! 3. Real-time Estimation Error & Filter Performance Telemetry.

use aegis_filter::{FilterError, ScalarKalmanFilter};

fn main() {
    println!("===============================================================================");
    println!("  AEGIS-FILTER :: FLIGHT TELEMETRY & ADVERSARIAL RESILIENCE SIMULATION");
    println!("  Developed under Exploidus (https://github.com/Exploidus)");
    println!("===============================================================================\n");

    // Flight Profile:
    // Drone climbing from ground (0m) to cruising altitude (50m) at 1.0 m/s
    let mut filter = ScalarKalmanFilter::new(
        0.0,  // Initial altitude: 0.0 meters
        5.0,  // Initial variance: high uncertainty
        0.02, // Process noise (Q): small aerodynamic disturbances
        0.8,  // Measurement noise (R): standard barometric sensor noise
    )
    .expect("Failed to initialize filter");

    let total_steps = 30;
    let dt = 1.0; // 1 second intervals
    let climb_rate = 1.5; // m/s

    println!(
        "{:<6} | {:<12} | {:<14} | {:<14} | {:<12} | {:<10}",
        "Step", "True Alt (m)", "Sensor Raw (m)", "Aegis Est (m)", "Abs Error", "Status"
    );
    println!(
        "{:-<6}-+-{:-<12}-+-{:-<14}-+-{:-<14}-+-{:-<12}-+-{:-<10}",
        "", "", "", "", "", ""
    );

    let mut accepted_count = 0;
    let mut rejected_count = 0;
    let mut unexpected_error_count = 0;
    let total_spikes = 2; // Injected at step 12 (+150m spoof) and step 22 (-80m glitch)

    for step in 1..=total_steps {
        let true_altitude = step as f64 * climb_rate;

        // Predict state forward
        filter.predict(climb_rate * dt).unwrap();

        // Introduce simulated sensor noise and occasional adversarial spoofing spikes
        let raw_measurement = if step == 12 {
            // Simulated GPS/Baro Spoofing Attack (+150m instant jump)
            true_altitude + 150.0
        } else if step == 22 {
            // Severe sensor hardware glitch (-80m drop)
            true_altitude - 80.0
        } else {
            // Normal Gaussian-like noise (sine oscillation pseudo-noise)
            let noise = ((step as f64 * 1.7).sin() * 0.7) + ((step as f64 * 3.1).cos() * 0.4);
            true_altitude + noise
        };

        // 3-Sigma Chi-square Outlier Rejection Gating
        let (estimated, status) = match filter.update(raw_measurement, Some(3.0)) {
            Ok(est) => {
                accepted_count += 1;
                (est, "VALID")
            }
            Err(FilterError::MeasurementGated) => {
                rejected_count += 1;
                // On rejected anomaly, state continues safely from prediction
                (filter.state(), "SPOOF REJECT")
            }
            Err(FilterError::NumericalInstability) => {
                unexpected_error_count += 1;
                eprintln!("[ERROR] Numerical instability detected at step {}", step);
                (filter.state(), "NUM_INSTAB")
            }
            Err(FilterError::SingularInnovation) => {
                unexpected_error_count += 1;
                eprintln!("[ERROR] Singular innovation variance at step {}", step);
                (filter.state(), "SINGULAR")
            }
            Err(e) => {
                unexpected_error_count += 1;
                eprintln!("[ERROR] Filter error at step {}: {:?}", step, e);
                (filter.state(), "ERROR")
            }
        };

        let abs_error = (estimated - true_altitude).abs();

        println!(
            "{:<6} | {:<12.2} | {:<14.2} | {:<14.2} | {:<12.3} | {:<10}",
            step, true_altitude, raw_measurement, estimated, abs_error, status
        );
    }

    let spike_block_pct = (rejected_count as f64 / total_spikes as f64) * 100.0;
    let total_gated_pct = (rejected_count as f64 / total_steps as f64) * 100.0;

    println!("\n-------------------------------------------------------------------------------");
    println!("  SIMULATION SUMMARY:");
    println!("  - Total Steps Processed      : {}", total_steps);
    println!("  - Valid Readings Fused       : {}", accepted_count);
    println!(
        "  - Adversarial Spikes Blocked : {} / {} ({:.1}% blocked, {:.1}% of total stream gated)",
        rejected_count, total_spikes, spike_block_pct, total_gated_pct
    );
    println!(
        "  - Unexpected Filter Errors   : {}",
        unexpected_error_count
    );
    println!("  - Final Variance Confidence  : {:.4}", filter.variance());
    println!("===============================================================================");

    assert_eq!(
        unexpected_error_count, 0,
        "No unexpected numerical or singular filter errors should occur"
    );
}
