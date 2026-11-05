#[cfg(test)]
mod h_infinity {
    use aegis_filter::{FilterError, HInfinityFilter1D};

    #[test]
    fn robustness() {
        let mut h_inf = HInfinityFilter1D::new(10.0, 1.0, 0.05, 0.5, 5.0).unwrap();

        for _ in 0..20 {
            h_inf.predict(0.0).unwrap();
            let _ = h_inf.update(10.1).unwrap();
        }

        assert!((h_inf.state() - 10.1).abs() < 0.3);
        assert!(h_inf.bound() > 0.0);
    }

    #[test]
    fn known_reference_gain() {
        // Analytical benchmark test pinning Simon (2006, Eq. 11.45-11.48)
        // Given: x0 = 0.0, P0 = 1.0, Q = 0.0, R = 0.5, gamma = 5.0
        // Riccati Denominator: D = 1 - P/gamma^2 + P/R = 1 - 1/25 + 1/0.5 = 2.96
        // Minimax Gain: K = P / (R * D) = 1.0 / (0.5 * 2.96) = 1.0 / 1.48
        // When measurement z = 1.48, delta = K * residual = (1.0 / 1.48) * 1.48 = 1.0 exactly.
        let mut h_inf = HInfinityFilter1D::new(0.0, 1.0, 0.0, 0.5, 5.0).unwrap();
        let new_state = h_inf.update(1.48).unwrap();
        assert!(
            (new_state - 1.0).abs() < 1e-12,
            "Expected exactly 1.0, got {}",
            new_state
        );
        let expected_post_cov = 1.0 / 2.96;
        assert!(
            (h_inf.bound() - expected_post_cov).abs() < 1e-12,
            "Expected post-update bound {}, got {}",
            expected_post_cov,
            h_inf.bound()
        );
    }

    #[test]
    fn riccati_instability_guard() {
        // When gamma bound is tighter than physically admissible under system noise:
        // P = 10.0, R = 1.0, gamma = 0.5 => gamma^2 = 0.25
        // D = 1 - 10/0.25 + 10/1 = 1 - 40 + 10 = -29 <= 0
        let mut h_inf = HInfinityFilter1D::new(0.0, 10.0, 0.0, 1.0, 0.5).unwrap();
        let err = h_inf.update(1.0);
        assert_eq!(err, Err(FilterError::NumericalInstability));
    }
}
