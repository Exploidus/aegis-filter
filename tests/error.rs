#[cfg(test)]
mod err {
    use aegis_filter::{
        AdaptiveFilter1D, FilterError, HInfinityFilter1D, MultiSensorFusion1D, Real,
        ScalarKalmanFilter, SensorSource,
    };

    #[test]
    fn capacity_exceeded() {
        let mut fusion = MultiSensorFusion1D::<2>::new(0.0, 1.0, 0.01).unwrap();
        let s1 = SensorSource::new(1, 0.2, 3).unwrap();
        let s2 = SensorSource::new(2, 0.3, 3).unwrap();
        let s3 = SensorSource::new(3, 0.4, 3).unwrap();

        assert_eq!(fusion.register_sensor(s1), Ok(()));
        assert_eq!(fusion.register_sensor(s2), Ok(()));
        // Array capacity is 2; 3rd sensor must return CapacityExceeded
        assert_eq!(
            fusion.register_sensor(s3),
            Err(FilterError::CapacityExceeded)
        );
    }

    #[test]
    fn invalid_parameter() {
        assert_eq!(
            SensorSource::new(1, 0.0, 3).err(),
            Some(FilterError::InvalidParameter)
        );
        assert_eq!(
            SensorSource::new(1, -0.5, 3).err(),
            Some(FilterError::InvalidParameter)
        );
        let s = SensorSource::new(1, 0.5, 3).unwrap();
        assert_eq!(
            s.with_gate_threshold(0.0).err(),
            Some(FilterError::InvalidParameter)
        );
        assert_eq!(
            MultiSensorFusion1D::<1>::new(0.0, -1.0, 0.01).err(),
            Some(FilterError::InvalidParameter)
        );
    }

    #[test]
    fn invalid_covariance() {
        assert_eq!(
            ScalarKalmanFilter::new(0.0, 0.0, 0.01, 0.5).err(),
            Some(FilterError::InvalidCovariance)
        );
        assert_eq!(
            ScalarKalmanFilter::new(0.0, -1.0, 0.01, 0.5).err(),
            Some(FilterError::InvalidCovariance)
        );
        assert_eq!(
            HInfinityFilter1D::new(0.0, 1.0, 0.01, 0.5, 0.0).err(),
            Some(FilterError::InvalidCovariance)
        );
        assert_eq!(
            AdaptiveFilter1D::new(0.0, 1.0, 0.01, 0.5, 1.5).err(),
            Some(FilterError::InvalidCovariance)
        );
    }

    #[test]
    fn zero_panics_property_stress() {
        let extreme_floats = [
            Real::NAN,
            Real::INFINITY,
            Real::NEG_INFINITY,
            0.0,
            -0.0,
            Real::MIN_POSITIVE,
            Real::MAX,
            -Real::MAX,
            #[cfg(not(feature = "single-precision"))]
            1e-300,
            #[cfg(not(feature = "single-precision"))]
            1e300,
            #[cfg(feature = "single-precision")]
            1e-35,
            #[cfg(feature = "single-precision")]
            1e35,
        ];

        let mut kf = ScalarKalmanFilter::new(0.0, 1.0, 0.01, 0.5).unwrap();
        let mut hinf = HInfinityFilter1D::new(0.0, 1.0, 0.01, 0.5, 5.0).unwrap();
        let mut af = AdaptiveFilter1D::new(0.0, 1.0, 0.01, 0.5, 0.05).unwrap();

        for &val in &extreme_floats {
            // Predict calls must never panic
            let _ = kf.predict(val);
            let _ = hinf.predict(val);
            let _ = af.predict(val);

            // Update calls must return Result, never panic
            let _ = kf.update(val, Some(3.0));
            let _ = hinf.update(val);
            let _ = af.update(val);
        }
    }
}
