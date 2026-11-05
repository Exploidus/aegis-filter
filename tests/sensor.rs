#[cfg(test)]
mod sensor {
    use aegis_filter::{FusionStatus, MultiSensorFusion1D, SensorMeasurement, SensorSource};
    #[test]
    fn multi_fusion_and_quarantine() {
        // Setup fusion engine with capacity for 3 sensors
        let mut fusion = MultiSensorFusion1D::<3>::new(0.0, 5.0, 0.05).unwrap();

        // Sensor 1: Accurate Barometer (R=0.2, max 3 anomalies)
        let baro = SensorSource::new(1, 0.2, 3).unwrap();
        // Sensor 2: Noisy GPS (R=1.5, max 2 anomalies)
        let gps = SensorSource::new(2, 1.5, 2).unwrap();

        fusion.register_sensor(baro).unwrap();
        fusion.register_sensor(gps).unwrap();

        // Nominal fusion
        fusion.predict(1.0).unwrap();
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 1.05,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Fused);

        // Adversarial spoofing on GPS (id=2)
        // Attack 1: Gated
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 2,
                value: 500.0,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Gated);

        // Attack 2: Second consecutive anomaly triggers Quarantine!
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 2,
                value: 500.0,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Quarantined);
        assert_eq!(fusion.sensor_health(2), Some(0.0));

        // Subsequent measurements from quarantined GPS are safely ignored
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 2,
                value: 1.0,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::IgnoredQuarantined);

        // Barometer continues nominal operation unhindered
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 1.02,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Fused);
    }

    #[test]
    fn auto_recovery() {
        let mut fusion = MultiSensorFusion1D::<2>::new(10.0, 1.0, 0.01).unwrap();
        // Sensor with auto-recovery threshold of 2 consecutive healthy observations
        let gps = SensorSource::new(1, 0.5, 2).unwrap().with_auto_recovery(2);
        fusion.register_sensor(gps).unwrap();

        // 2 anomalies force quarantine
        let _ = fusion.fuse(SensorMeasurement {
            sensor_id: 1,
            value: 100.0,
        });
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 100.0,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Quarantined);

        // 1st healthy sample while quarantined
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 10.05,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::IgnoredQuarantined);

        // 2nd healthy sample triggers auto-recovery!
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 10.02,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Recovered);

        // Subsequent sample is actively fused again
        let status = fusion
            .fuse(SensorMeasurement {
                sensor_id: 1,
                value: 10.01,
            })
            .unwrap();
        assert_eq!(status, FusionStatus::Fused);
    }
}
