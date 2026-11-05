#[cfg(test)]
mod adaptive_filter {
    use aegis_filter::AdaptiveFilter1D;
    #[test]
    fn noise_tracking() {
        let mut af = AdaptiveFilter1D::new(0.0, 5.0, 0.01, 1.0, 0.05).unwrap();
        for _ in 0..40 {
            af.predict(0.0).unwrap();
            let _ = af.update(20.0).unwrap();
        }
        assert!((af.x - 20.0).abs() < 0.5, "af.x = {}", af.x);
    }

    #[test]
    fn bounds_and_prior_covariance() {
        let mut af = AdaptiveFilter1D::new(0.0, 2.0, 0.01, 1.0, 0.1).unwrap();
        // Severe adversarial jump: residual is 1000.0, y^2 = 1e6
        af.predict(0.0).unwrap();
        let _ = af.update(1000.0).unwrap();
        // Online R estimate must be clamped within upper bound (<= 1e6)
        assert!(af.estimated_measurement_noise() <= 1e6);
        assert!(af.estimated_measurement_noise() >= 1e-6);

        // Sub-noise tiny measurement where y^2 < P_prior:
        for _ in 0..20 {
            af.predict(0.0).unwrap();
            let _ = af.update(af.x).unwrap();
        }
        // Online R estimate must stay strictly positive, never collapsing below 1e-6
        assert!(af.estimated_measurement_noise() >= 1e-6);
    }
}
