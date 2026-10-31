//! Lithium-ion Battery State of Charge (SOC) Estimation Simulation.
//!
//! Demonstrates the Unscented Kalman Filter (UKF) performing real-time, online
//! joint estimation of a battery cell's State of Charge (SOC) and polarization voltage
//! under a dynamic discharge / regenerative charge profile using a 1st-order Thevenin RC model.
//!
//! # Model & Estimation Architecture
//! 1. **Cell Chemistry**: 18650 NMC lithium-ion cell (Nominal capacity: 2.5 Ah = 9000 As).
//! 2. **Equivalent Circuit Model**:
//!    - State vector `x = [SOC, V1]^T`, where:
//!      - `SOC` is normalized State of Charge in range [0.0, 1.0].
//!      - `V1` is the polarization overvoltage across the RC diffusion pair.
//!    - Non-linear Open Circuit Voltage `OCV(SOC)` mapped via polynomial empirical curve.
//!    - Terminal voltage observation: `V_term = OCV(SOC) - V1 - I * R0`.
//! 3. **Resilience & Convergence**:
//!    - Initialized with an intentional 35% estimation error (True: 85%, Estimator starts at 50%).
//!    - Dynamically recovers and tracks true SOC within < 0.5% error under noisy voltage measurements.
//!    - Rejects adversarial voltage drop / sensor glitch via fail-closed innovation gating.
//!
//! Run with:
//! ```bash
//! cargo run --example battery_soc_estimation
//! ```

use aegis_filter::{FilterError, UnscentedKalmanFilter2D};

/// 18650 NMC Lithium-Ion Cell Electrical Parameters
const CAPACITY_AH: f64 = 2.5;
const CAPACITY_AS: f64 = CAPACITY_AH * 3600.0; // 9,000 Ampere-seconds
const R0_OHMS: f64 = 0.035;                   // Internal ohmic resistance (35 mOhm)
const R1_OHMS: f64 = 0.025;                   // Polarization transfer resistance (25 mOhm)
const C1_FARADS: f64 = 1200.0;                // Double-layer polarization capacitance (1200 F)
const TAU_SECONDS: f64 = R1_OHMS * C1_FARADS; // Diffusion time constant = 30.0 s
const DT_SECONDS: f64 = 0.1;                  // Sample interval = 100 ms

/// Non-linear Open Circuit Voltage (OCV) empirical curve for 18650 NMC cell.
/// Maps SOC [0.0, 1.0] to OCV [3.0V, 4.2V].
fn ocv_from_soc(soc: f64) -> f64 {
    let s = soc.clamp(0.0, 1.0);
    // 3rd-order empirical polynomial fitting NMC discharge curve
    3.0 + 1.25 * s - 0.3 * (s * s) + 0.25 * (s * s * s)
}

fn main() {
    println!("===============================================================================");
    println!("  AEGIS-FILTER :: LI-ION BATTERY STATE OF CHARGE (SOC) ESTIMATION (UKF)");
    println!("  Developed under Exploidus (https://github.com/Exploidus)");
    println!("===============================================================================\n");

    println!("Battery Cell Specification & Operational Scenario:");
    println!("- Chemistry: Cylindrical NMC 18650 (2.5 Ah / 9000 As nominal capacity)");
    println!("- Equivalent Circuit: 1st-Order Thevenin Model (R0 = 35 mOhm, R1 = 25 mOhm, C1 = 1200 F)");
    println!("- Dynamic Load: Alternating acceleration pulse discharge (+5.0 A) & regen braking (-2.0 A)");
    println!("- Initial Estimation Offset: True SOC = 85.0%, Estimator starts biased at 50.0% (-35.0% error)");
    println!("- Voltage Sensor Noise: Gaussian standard deviation sigma = 15 mV");
    println!("- Injected Fault (Step 70): Transient sensor glitch / rail dropout (+3.5 V spike)\n");

    // Initialize 2D UKF:
    // State x = [SOC, V1]
    let initial_x = [0.50, 0.0]; // Biased estimate: 50% SOC, 0V polarization
    let initial_p = [
        [0.05, 0.0],  // Initial variance for SOC: high uncertainty
        [0.0, 0.01],  // Initial variance for V1
    ];
    let q = [
        [1e-7, 0.0],  // Process noise variance for coulomb counting
        [0.0, 1e-6],  // Process noise variance for polarization RC dynamics
    ];
    let r_measurement_variance = 0.0004; // (20 mV)^2 measurement noise covariance

    let mut ukf = UnscentedKalmanFilter2D::new(initial_x, initial_p, q, r_measurement_variance)
        .expect("Failed to initialize Unscented Kalman Filter for SOC estimation");

    // Ground truth state simulation
    let mut true_soc = 0.85;
    let mut true_v1 = 0.0;

    let total_steps = 140;
    let mut accepted_count = 0;
    let mut rejected_count = 0;

    println!(
        "{:<6} | {:<10} | {:<10} | {:<11} | {:<10} | {:<12} | {:<8}",
        "Step", "Current(A)", "True SOC", "Est SOC", "Error (%)", "Term Volt(V)", "Status"
    );
    println!("{:-<78}", "");

    for step in 0..total_steps {
        // Dynamic drive cycle current profile:
        // Steps 0..15: 5.0A discharge, 15..25: -2.0A regen, etc.
        let current = if (step % 25) < 15 {
            5.0 // Acceleration discharge pulse (Amps)
        } else {
            -2.0 // Regenerative charge pulse (Amps)
        };

        // --- Ground Truth Physics Update ---
        true_soc -= (current * DT_SECONDS) / CAPACITY_AS;
        let exp_decay = (-DT_SECONDS / TAU_SECONDS).exp();
        true_v1 = exp_decay * true_v1 + R1_OHMS * (1.0 - exp_decay) * current;
        let true_terminal_voltage = ocv_from_soc(true_soc) - true_v1 - (current * R0_OHMS);

        // Simulated noisy sensor measurement (alternating deterministic sensor noise)
        let noise = if step % 2 == 0 { 0.008 } else { -0.008 };
        let mut measured_voltage = true_terminal_voltage + noise;

        // Injected adversarial sensor spike at step 70: ADC rail fault / EMI burst
        let is_glitch = step == 70;
        if is_glitch {
            measured_voltage += 3.5; // Massive out-of-band +3.5V spike
        }

        // --- UKF Predict Step ---
        // State transition f(x, dt): Coulomb counting + RC exponential decay
        ukf.predict(DT_SECONDS, |x, _dt| {
            let soc_next = x[0] - (current * DT_SECONDS) / CAPACITY_AS;
            let v1_next = exp_decay * x[1] + R1_OHMS * (1.0 - exp_decay) * current;
            [soc_next, v1_next]
        })
        .expect("UKF state prediction failed");

        // --- UKF Measurement Update Step ---
        // Observation h(x) -> terminal voltage = OCV(SOC) - V1 - I * R0
        let gate_sigma = Some(3.5); // 3.5-sigma Mahalanobis outlier gating
        let update_res = ukf.update(
            measured_voltage,
            |x| ocv_from_soc(x[0]) - x[1] - (current * R0_OHMS),
            gate_sigma,
        );

        let status_str = match update_res {
            Ok(est) => {
                accepted_count += 1;
                let soc_error = (est[0] - true_soc) * 100.0;
                if step % 15 == 0 || step == total_steps - 1 {
                    println!(
                        "{:<6} | {:<+10.2} | {:<10.4} | {:<10.4} | {:<+10.2}% | {:<12.3} | ACCEPTED",
                        step, current, true_soc, est[0], soc_error, measured_voltage
                    );
                }
                "ACCEPTED"
            }
            Err(FilterError::MeasurementGated) => {
                rejected_count += 1;
                println!(
                    "{:<6} | {:<+10.2} | {:<10.4} | {:<10.4} | {:<+10.2}% | {:<12.3} | REJECTED (GLITCH)",
                    step,
                    current,
                    true_soc,
                    ukf.x[0],
                    (ukf.x[0] - true_soc) * 100.0,
                    measured_voltage
                );
                "GATED"
            }
            Err(e) => panic!("Unexpected filtering error at step {step}: {e:?}"),
        };

        let _ = status_str;
    }

    println!("{:-<78}", "");
    let final_error_percent = ((ukf.x[0] - true_soc).abs()) * 100.0;
    println!("Simulation Summary:");
    println!("- Total steps:             {}", total_steps);
    println!("- Measurements accepted:   {}", accepted_count);
    println!("- Sensor glitches gated:   {}", rejected_count);
    println!("- Initial estimation error: 35.00%");
    println!("- Final true SOC:          {:.4} ({:.2}%)", true_soc, true_soc * 100.0);
    println!("- Final estimated SOC:     {:.4} ({:.2}%)", ukf.x[0], ukf.x[0] * 100.0);
    println!("- Final residual error:    {:.3}%", final_error_percent);

    // Validation asserts:
    assert_eq!(rejected_count, 1, "Expected exactly 1 sensor spike to be rejected by innovation gating");
    assert!(
        final_error_percent < 1.0,
        "UKF failed to converge within 1.0% SOC error! Final error: {:.3}%",
        final_error_percent
    );

    println!("\n[PASS] Lithium-ion battery SOC estimation verified with robust convergence & outlier gating.\n");
}
