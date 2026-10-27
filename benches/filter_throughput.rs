use aegis_filter::{HInfinityFilter1D, ScalarKalmanFilter, UnscentedKalmanFilter2D};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_scalar_kalman(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_kalman");

    group.bench_function("predict", |b| {
        let mut filter = ScalarKalmanFilter::new(0.0, 1.0, 0.01, 0.5).unwrap();
        b.iter(|| {
            filter.predict(black_box(0.01)).unwrap();
        });
    });

    group.bench_function("update", |b| {
        let mut filter = ScalarKalmanFilter::new(0.0, 1.0, 0.01, 0.5).unwrap();
        let mut z = 10.0;
        b.iter(|| {
            z += 0.001;
            black_box(filter.update(black_box(z), None).unwrap());
        });
    });

    group.bench_function("predict_and_update", |b| {
        let mut filter = ScalarKalmanFilter::new(0.0, 1.0, 0.01, 0.5).unwrap();
        let mut z = 10.0;
        b.iter(|| {
            filter.predict(black_box(0.01)).unwrap();
            z += 0.001;
            black_box(filter.update(black_box(z), None).unwrap());
        });
    });

    group.finish();
}

fn bench_hinfinity(c: &mut Criterion) {
    let mut group = c.benchmark_group("hinfinity_1d");

    group.bench_function("predict", |b| {
        let mut filter = HInfinityFilter1D::new(0.0, 1.0, 0.01, 0.5, 5.0).unwrap();
        b.iter(|| {
            filter.predict(black_box(0.01)).unwrap();
        });
    });

    group.bench_function("update", |b| {
        let mut filter = HInfinityFilter1D::new(0.0, 1.0, 0.01, 0.5, 5.0).unwrap();
        let mut z = 10.0;
        b.iter(|| {
            z += 0.001;
            black_box(filter.update(black_box(z)).unwrap());
        });
    });

    group.bench_function("predict_and_update", |b| {
        let mut filter = HInfinityFilter1D::new(0.0, 1.0, 0.01, 0.5, 5.0).unwrap();
        let mut z = 10.0;
        b.iter(|| {
            filter.predict(black_box(0.01)).unwrap();
            z += 0.001;
            black_box(filter.update(black_box(z)).unwrap());
        });
    });

    group.finish();
}

fn bench_ukf_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("unscented_kalman_2d");

    group.bench_function("predict_and_update", |b| {
        let mut ukf = UnscentedKalmanFilter2D::new(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.01, 0.0], [0.0, 0.01]],
            0.5,
        )
        .unwrap();
        b.iter(|| {
            ukf.predict(black_box(0.01), |x, _dt| [x[0] + 0.01, x[1]])
                .unwrap();
            black_box(ukf.update(black_box(1.0), |x| x[0], None).unwrap());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_scalar_kalman, bench_hinfinity, bench_ukf_2d);
criterion_main!(benches);
