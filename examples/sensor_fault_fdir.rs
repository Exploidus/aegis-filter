//! Multi-sensor fusion and real-time Fault Detection, Isolation, and Recovery (FDIR) simulation.
//!
//! Simulates an autonomous aircraft fusing redundant sensors:
//! 1. Barometric Altimeter (High frequency, lower noise)
//! 2. GPS Receiver (Medium frequency, subject to active adversarial RF spoofing)
//! 3. Ground Radar/LiDAR (Accurate close to ground, drops out higher up)

use aegis_filter::{MultiSensorFusion1D, SensorMeasurement, SensorSource};

fn main() {
    println!("===============================================================================");
    println!("  AEGIS-FILTER :: MULTI-SENSOR FUSION & DYNAMIC FAULT ISOLATION (FDIR)");
    println!("  Developed under Exploidus (https://github.com/Exploidus)");
    println!("===============================================================================\n");

    // Initialize fusion engine with capacity for 4 redundant sensors
    let mut fusion = MultiSensorFusion1D::<4>::new(0.0, 5.0, 0.02)
        .expect("Failed to initialize fusion filter");

    // Register sensors:
    // Sensor 1: Barometer (Variance R = 0.3, Max 4 consecutive anomalies)
    let baro = SensorSource::new(1, 0.3, 4).expect("Failed to create Baro config");
    // Sensor 2: GPS Altimeter (Variance R = 1.2, Max 2 consecutive anomalies)
    let gps = SensorSource::new(2, 1.2, 2).expect("Failed to create GPS config");
    // Sensor 3: LiDAR Altimeter (Variance R = 0.05, Max 3 consecutive anomalies)
    let lidar = SensorSource::new(3, 0.05, 3).expect("Failed to create LiDAR config");

    fusion.register_sensor(baro).unwrap();
    fusion.register_sensor(gps).unwrap();
    fusion.register_sensor(lidar).unwrap();

    let total_steps = 20;
    let climb_rate = 2.0;

    println!(
        "{:<5} | {:<10} | {:<8} | {:<12} | {:<14} | {:<18}",
        "Step", "True (m)", "Sensor", "Sensor (m)", "Fused Est (m)", "Status"
    );
    println!(
        "{}-+-{}-+-{}-+-{}-+-{}-+-{}",
        "-".repeat(5),
        "-".repeat(10),
        "-".repeat(8),
        "-".repeat(12),
        "-".repeat(14),
        "-".repeat(18)
    );

    for step in 1..=total_steps {
        let true_alt = (step as f64) * climb_rate;
        fusion.predict(climb_rate).unwrap();

        // 1. Barometer measurement (nominal with small noise)
        let baro_reading = true_alt + 0.15;
        let s1 = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: baro_reading,
            })
            .unwrap();

        println!(
            "{:<5} | {:<10.2} | {:<8} | {:<12.2} | {:<14.2} | {:?}",
            step, true_alt, "Baro", baro_reading, fusion.state(), s1
        );

        // 2. GPS measurement (injected with an adversarial RF spoofing attack at step 8 & 9)
        let gps_reading = if step == 8 || step == 9 {
            true_alt + 250.0 // Spoofed spike
        } else {
            true_alt - 0.25
        };

        let s2 = fusion
            .fuse(SensorMeasurement {
                sensor_id: 2,
                value: gps_reading,
            })
            .unwrap();

        println!(
            "{:<5} | {:<10.2} | {:<8} | {:<12.2} | {:<14.2} | {:?}",
            step, true_alt, "GPS", gps_reading, fusion.state(), s2
        );

        // 3. LiDAR measurement (valid only below 25m)
        if true_alt <= 25.0 {
            let lidar_reading = true_alt + 0.02;
            let s3 = fusion
                .fuse(SensorMeasurement {
                    sensor_id: 3,
                    value: lidar_reading,
                })
                .unwrap();

            println!(
                "{:<5} | {:<10.2} | {:<8} | {:<12.2} | {:<14.2} | {:?}",
                step, true_alt, "LiDAR", lidar_reading, fusion.state(), s3
            );
        }
    }

    println!("\n-------------------------------------------------------------------------------");
    println!("  FINAL FAULT DIAGNOSTICS & SENSOR STATUS:");
    println!(
        "  - Barometer (ID 1) Health : {:.1}% (Active)",
        fusion.sensor_health(1).unwrap() * 100.0
    );
    println!(
        "  - GPS       (ID 2) Health : {:.1}% (QUARANTINED)",
        fusion.sensor_health(2).unwrap() * 100.0
    );
    println!(
        "  - LiDAR     (ID 3) Health : {:.1}% (Active)",
        fusion.sensor_health(3).unwrap() * 100.0
    );
    println!("  - Final Variance P        : {:.6}", fusion.variance());
    println!("===============================================================================");
}
