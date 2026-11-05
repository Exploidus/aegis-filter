#[cfg(test)]
mod scalar {
    use aegis_filter::{FilterError, Real, ScalarKalmanFilter};
    #[cfg(test)]
    mod filter {
        use super::*;
        #[test]
        fn filter_convergence() {
            let mut filter = ScalarKalmanFilter::new(0.0, 10.0, 0.01, 0.5).unwrap();
            let target = 25.0;
            for _ in 0..50 {
                filter.predict(0.0).unwrap();
                let _ = filter.update(target + 0.05, None).unwrap();
            }
            assert!((filter.state() - target).abs() < 0.2);
        }

        #[test]
        fn filter_outlier_gating() {
            let mut filter = ScalarKalmanFilter::new(10.0, 1.0, 0.01, 0.5).unwrap();
            // Adversarial spike — must be rejected by 3-sigma gating
            let err = filter.update(9999.0, Some(3.0));
            assert_eq!(err, Err(FilterError::MeasurementGated));
            assert!((filter.state() - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn nan_inf_rejection() {
        let result = ScalarKalmanFilter::new(Real::NAN, 1.0, 0.1, 1.0);
        assert_eq!(result.err(), Some(FilterError::NonFiniteInput));

        let mut f = ScalarKalmanFilter::new(0.0, 1.0, 0.1, 1.0).unwrap();
        assert_eq!(
            f.update(Real::INFINITY, None).err(),
            Some(FilterError::NonFiniteInput)
        );
    }
}
