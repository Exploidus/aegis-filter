//! Non-linear Radar Tracking & Bearing Estimation Simulation.
//!
//! Demonstrates the Unscented Kalman Filter (UKF) tracking a target whose
//! motion is propagated by a linear constant-velocity transition and observed
//! through a NON-LINEAR sensor: the range `h(x) = sqrt(x0^2 + x1^2)`. That
//! non-linear observation (no analytical Jacobian) is exactly the case the UKF
//! exists for.
//!
//! # Why this version (fixes #3)
//! The original demo froze the y-state in the transition while the "true" y
//! drifted, and read a single range off a 2-D state. A single range cannot
//! separate x from y, so the estimator was fighting a model that did not match
//! reality and the "tracked" line was printed unconditionally even while the
//! error grew. The fix is three-fold, each commented at its site below:
//!   1. Propagate the target consistently with the ground truth.
//!   2. Add a second, observable sensor (a lateral/y reading) so both states
//!      can actually be recovered — range alone is unobservable in the direction
//!      orthogonal to the line of sight.
//!   3. Report "tracked" only when the final error is actually small (guarded by
//!      an assertion so a regression cannot print a false success again).

use aegis_filter::UnscentedKalmanFilter2D;

// Target flies a straight line at constant velocity: x = x0 + vx*t, y = y0 + vy*t.
// A linear transition model can only track linear motion, so the "curved path"
// in the old demo was the first inconsistency between model and ground truth.
const VX: f64 = 8.0;
const VY: f64 = 3.0;

// Final-error under which we honestly call the run a "track".
const CONVERGENCE_THRESHOLD_M: f64 = 10.0;

fn main() {
    println!("===============================================================================");
    println!("  AEGIS-FILTER :: NON-LINEAR RADAR TRACKING SIMULATION (UKF)");
    println!("  Developed under Exploidus (https://github.com/Exploidus)");
    println!("===============================================================================\n");

    // Initial state: target at (100m, 100m).
    let mut ukf = UnscentedKalmanFilter2D::new(
        [100.0, 100.0],
        [[10.0, 0.0], [0.0, 10.0]], // Initial state covariance
        [[0.2, 0.0], [0.0, 0.2]],   // Process noise Q (constant-velocity model)
        2.0,                        // Radar range measurement noise R
    )
    .expect("Failed to initialize UKF");

    let total_steps = 15;
    let dt = 1.0;

    let mut final_error = 0.0_f64;

    println!(
        "{:<6} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>8}",
        "Step", "TruX", "TruY", "Range", "LatY", "EstX", "EstY", "Err(m)"
    );
    println!(
        "{:-<6}-+-{:-<7}-+-{:-<7}-+-{:-<7}-+-{:-<7}-+-{:-<7}-+-{:-<7}-+-{:-<8}",
        "", "", "", "", "", "", "", ""
    );

    for step in 1..=total_steps {
        // Ground truth: constant-velocity straight-line motion.
        let true_x = 100.0 + VX * step as f64;
        let true_y = 100.0 + VY * step as f64;

        // (1) Model the SAME motion the target actually makes. Capturing vx, vy in
        // the transition closure propagates x0 *and* x1, so the predictor no longer
        // freezes y the way the old demo did.
        ukf.predict(dt, |state, delta| {
            [state[0] + VX * delta, state[1] + VY * delta]
        })
        .unwrap();

        // (2) Fuse TWO measurements each step: the non-linear range and a lateral
        // y reading. Together they make the 2-D state observable; range alone could
        // not separate x from y.
        let range = (true_x * true_x + true_y * true_y).sqrt();
        let noisy_range = range + 0.3; // small deterministic sensor offset
        ukf.update(
            noisy_range,
            |state| (state[0] * state[0] + state[1] * state[1]).sqrt(),
            None,
        )
        .unwrap();

        let noisy_lateral = true_y + 0.1;
        let estimated = ukf.update(noisy_lateral, |state| state[1], None).unwrap();

        let dx = estimated[0] - true_x;
        let dy = estimated[1] - true_y;
        let pos_error = (dx * dx + dy * dy).sqrt();
        final_error = pos_error;

        println!(
            "{:<6} | {:>7.1} | {:>7.1} | {:>7.2} | {:>7.2} | {:>7.1} | {:>7.1} | {:>8.2}",
            step, true_x, true_y, noisy_range, noisy_lateral, estimated[0], estimated[1], pos_error
        );
    }

    println!("\n-------------------------------------------------------------------------------");
    // (3) Only claim a track when the error is actually small. The assert is a
    // regression guard: if a future change makes the demo diverge again, the
    // example panics instead of silently printing a false success.
    if final_error <= CONVERGENCE_THRESHOLD_M {
        println!(
            "  UKF tracked a constant-velocity target through a non-linear \
             range observation to a final position error of {:.1} m.",
            final_error
        );
    } else {
        println!(
            "  Tracker DID NOT converge: final position error {:.1} m exceeds the \
             {:.1} m threshold.",
            final_error, CONVERGENCE_THRESHOLD_M
        );
    }
    assert!(
        final_error <= CONVERGENCE_THRESHOLD_M,
        "regression guard: demo diverged (error {final_error:.1} m > {CONVERGENCE_THRESHOLD_M} m)"
    );
    println!("===============================================================================");
}
