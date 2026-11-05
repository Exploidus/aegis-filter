#[cfg(test)]
mod tests {
    use aegis_filter::WindowedStats;
    #[test]
    fn windowed_stats_rolling_metrics() {
        let mut stats = WindowedStats::<4>::new();
        assert!(stats.is_empty());
        assert_eq!(stats.len(), 0);

        stats.push(2.0).unwrap();
        stats.push(4.0).unwrap();
        assert_eq!(stats.len(), 2);
        assert!((stats.mean().unwrap() - 3.0).abs() < 1e-9);
        assert_eq!(stats.min(), Some(2.0));
        assert_eq!(stats.max(), Some(4.0));
        assert!((stats.peak_to_peak().unwrap() - 2.0).abs() < 1e-9);

        // Push more samples to exceed window capacity of 4
        stats.push(6.0).unwrap();
        stats.push(8.0).unwrap();
        assert_eq!(stats.len(), 4);
        assert!((stats.mean().unwrap() - 5.0).abs() < 1e-9);

        // Overwrite oldest (2.0) with 10.0 -> window has [10, 4, 6, 8], mean = 7.0
        stats.push(10.0).unwrap();
        assert_eq!(stats.len(), 4);
        assert!((stats.mean().unwrap() - 7.0).abs() < 1e-9);
        assert_eq!(stats.min(), Some(4.0));
        assert_eq!(stats.max(), Some(10.0));
        assert!(stats.variance().unwrap() > 0.0);
    }
}
