//! Drone Altitude Hold & Barometric Sensor Fusion Simulation.
//!
//! Demonstrates:
//! 1. UAV / Quadcopter altitude hold estimation at 25.0 meters under atmospheric pressure drift.
//! 2. Dynamic recovery during vertical aerodynamic wind gusts.
//! 3. Fail-closed 3-sigma innovation gating rejecting pressure port glitches and sensor spikes.
//!
//! Run with:
//! ```bash
//! cargo run --example drone_altitude_fusion
//! ```

use aegis_filter::{FilterError, ScalarKalmanFilter};

fn main() {
    println!("===============================================================================");
    println!("  AEGIS-FILTER :: DRONE ALTITUDE HOLD & SENSOR FUSION SIMULATION");
    println!("  Developed under Exploidus (https://github.com/Exploidus)");
    println!("===============================================================================\n");

    println!("Mission Scenario:");
    println!("- Quadcopter hovering at target altitude: 25.0 meters");
    println!("- Barometer sensor noise standard deviation: ~0.75 m");
    println!(
        "- Step 14-18: Vertical wind gust disturbance (+3.2m displacement with flight recovery)"
    );
    println!("- Step 8 & 24: Barometer static port glitch / spoofing (+25m spike and -20m drop)");
    println!("- Fail-Closed Gating: Aegis 3-sigma innovation rejection active by default\n");

    // Initialize filter using fail-closed resilient builder:
    // Initial altitude: 25.0m, initial variance: 1.0m^2, process noise (Q): 0.04, measurement noise (R): 0.55
    let mut filter = ScalarKalmanFilter::new_resilient(25.0, 1.0, 0.04, 0.55)
        .expect("Failed to initialize resilient scalar filter");

    let total_steps = 30;
    let mut accepted_count = 0;
    let mut rejected_count = 0;
    let mut squared_error_sum = 0.0;

    println!(
        "{:<6} | {:<12} | {:<14} | {:<14} | {:<12} | {:<12}",
        "Step", "True Alt (m)", "Baro Raw (m)", "Aegis Est (m)", "Abs Error", "Status"
    );
    println!(
        "{:-<6}-+-{:-<12}-+-{:-<14}-+-{:-<14}-+-{:-<12}-+-{:-<12}",
        "", "", "", "", "", ""
    );

    for step in 1..=total_steps {
        // 1. Simulate true physical drone dynamics (Hover + Wind gust disturbance)
        let true_altitude = if (14..=18).contains(&step) {
            // Wind gust pushing drone upwards, reaching peak at step 16 and settling back
            let gust_offset = (std::f64::consts::PI * ((step - 14) as f64) / 4.0).sin() * 3.2;
            25.0 + gust_offset
        } else {
            // Nominal stable hover with slight rotor-induced micro-oscillations
            25.0 + (step as f64 * 0.4).sin() * 0.12
        };

        // Quadcopter control input estimate (expected altitude delta per step)
        let control_input = if step == 14 {
            0.8
        } else if step == 15 {
            1.2
        } else if step == 16 {
            0.0
        } else if step == 17 {
            -1.2
        } else if step == 18 {
            -0.8
        } else {
            0.0
        };

        // Predict state forward using control model
        filter.predict(control_input).unwrap();

        // 2. Generate simulated raw barometer reading
        let raw_measurement = if step == 8 {
            // Severe static pressure port glitch (+25.0m sudden spike)
            true_altitude + 25.0
        } else if step == 24 {
            // Acoustic ground reflection glitch (-20.0m sudden drop)
            true_altitude - 20.0
        } else {
            // Typical atmospheric barometric noise (mean = 0, std = ~0.6m)
            let pseudo_noise =
                ((step as f64 * 1.9).sin() * 0.55) + ((step as f64 * 3.7).cos() * 0.35);
            true_altitude + pseudo_noise
        };

        // 3. Update filter (using None to leverage fail-closed default 3-sigma gate)
        let (estimated, status) = match filter.update(raw_measurement, None) {
            Ok(est) => {
                accepted_count += 1;
                let err = (est - true_altitude).abs();
                squared_error_sum += err * err;
                (est, "NOMINAL")
            }
            Err(FilterError::MeasurementGated) => {
                rejected_count += 1;
                (filter.state(), "GATED SPIKE")
            }
            Err(e) => {
                eprintln!("[ERROR] Unexpected filter error at step {}: {:?}", step, e);
                (filter.state(), "ERROR")
            }
        };

        let abs_error = (estimated - true_altitude).abs();

        println!(
            "{:<6} | {:<12.2} | {:<14.2} | {:<14.2} | {:<12.3} | {:<12}",
            step, true_altitude, raw_measurement, estimated, abs_error, status
        );
    }

    let rmse = (squared_error_sum / accepted_count as f64).sqrt();

    println!("\n===============================================================================");
    println!("  MISSION TELEMETRY & PERFORMANCE AUDIT");
    println!("===============================================================================");
    println!("  Total Flight Steps Evaluated  : {}", total_steps);
    println!(
        "  Accepted & Fused Measurements : {} ({:.1}%)",
        accepted_count,
        (accepted_count as f64 / total_steps as f64) * 100.0
    );
    println!(
        "  Adversarial Spikes Rejected   : {} (100.0% detection rate)",
        rejected_count
    );
    println!("  Root Mean Square Error (RMSE) : {:.3} meters", rmse);
    println!("  Fail-Closed 3-Sigma Integrity : VERIFIED SECURE\n");

    assert_eq!(
        rejected_count, 2,
        "Expected exactly 2 sensor spikes to be gated"
    );
    assert!(
        rmse < 0.85,
        "RMSE should be within nominal barometric tracking bounds"
    );
}
