//! Non-linear Radar Tracking & Bearing Estimation Simulation.
//!
//! Demonstrates the Unscented Kalman Filter (UKF) tracking an aircraft flying a
//! non-linear curved path using a distance/range sensor h(x) = sqrt(x0^2 + x1^2).

use aegis_filter::UnscentedKalmanFilter2D;

fn main() {
    println!("===============================================================================");
    println!("  AEGIS-FILTER :: NON-LINEAR RADAR TRACKING SIMULATION (UKF)");
    println!("  Developed under Exploidus (https://github.com/Exploidus)");
    println!("===============================================================================\n");

    // Initial state: Aircraft at (100m, 100m)
    let mut ukf = UnscentedKalmanFilter2D::new(
        [100.0, 100.0],
        [[5.0, 0.0], [0.0, 5.0]],   // Initial state covariance
        [[0.05, 0.0], [0.0, 0.05]], // Process noise Q
        1.5,                        // Radar measurement noise R
    )
    .expect("Failed to initialize UKF");

    let total_steps = 15;
    let dt = 1.0;

    println!(
        "{:<6} | {:<18} | {:<14} | {:<18} | {:<10}",
        "Step", "True Pos (x, y)", "Radar Range", "UKF Est (x, y)", "Error (m)"
    );
    println!(
        "{:-<6}-+-{:-<18}-+-{:-<14}-+-{:-<18}-+-{:-<10}",
        "", "", "", "", ""
    );

    for step in 1..=total_steps {
        // True motion: Aircraft flying curved arc
        let angle = (step as f64) * 0.08;
        let true_x = 100.0 + (step as f64 * 8.0);
        let true_y = 100.0 + (angle.sin() * 25.0);

        // UKF non-linear prediction step
        ukf.predict(dt, |state, _delta| [state[0] + 8.0 * _delta, state[1]])
            .unwrap();

        // Non-linear Radar Range measurement: r = sqrt(x^2 + y^2)
        let true_range = (true_x * true_x + true_y * true_y).sqrt();
        let noisy_range = true_range + 0.3; // Slight sensor noise

        // UKF update through non-linear observation function
        let estimated = ukf
            .update(
                noisy_range,
                |state| (state[0] * state[0] + state[1] * state[1]).sqrt(),
                None,
            )
            .unwrap();

        let dx = estimated[0] - true_x;
        let dy = estimated[1] - true_y;
        let pos_error = (dx * dx + dy * dy).sqrt();

        println!(
            "{:<6} | ({:>6.1}, {:>6.1})  | {:<14.2} | ({:>6.1}, {:>6.1})  | {:<10.2}",
            step, true_x, true_y, noisy_range, estimated[0], estimated[1], pos_error
        );
    }

    println!("\n-------------------------------------------------------------------------------");
    println!("  UKF tracked continuous curved flight without analytical Jacobian matrices.");
    println!("===============================================================================");
}
