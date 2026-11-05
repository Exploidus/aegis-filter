#[cfg(test)]
mod c_ffi {
    use aegis_filter::ffi::AegisStatusCode;
    use core::mem::MaybeUninit;

    #[test]
    fn scalar_filter() {
        use aegis_filter::ffi::{
            aegis_scalar_filter_get_state, aegis_scalar_filter_get_variance,
            aegis_scalar_filter_init, aegis_scalar_filter_predict,
            aegis_scalar_filter_set_default_gate, aegis_scalar_filter_update, AegisScalarFilter,
            AEGIS_GATE_DISABLE,
        };

        unsafe {
            // Null pointer safety
            assert_eq!(
                aegis_scalar_filter_init(core::ptr::null_mut(), 0.0, 1.0, 0.01, 0.5),
                AegisStatusCode::NullPointer
            );

            let mut filter = MaybeUninit::<AegisScalarFilter>::uninit();
            let status = aegis_scalar_filter_init(filter.as_mut_ptr(), 0.0, 1.0, 0.01, 0.5);
            assert_eq!(status, AegisStatusCode::Ok);
            let mut filter = filter.assume_init();

            let status = aegis_scalar_filter_predict(&mut filter, 0.1);
            assert_eq!(status, AegisStatusCode::Ok);

            let mut out = 0.0;
            let status = aegis_scalar_filter_update(&mut filter, 1.05, 3.0, &mut out);
            assert_eq!(status, AegisStatusCode::Ok);
            assert!(out > 0.0);
            assert!((aegis_scalar_filter_get_state(&filter) - out).abs() < 1e-12);
            assert!(aegis_scalar_filter_get_variance(&filter) > 0.0);

            // Rejection of adversary spike
            let status = aegis_scalar_filter_update(&mut filter, 9999.0, 3.0, &mut out);
            assert_eq!(status, AegisStatusCode::MeasurementGated);

            // Test fail-closed default gate configuration
            let status = aegis_scalar_filter_set_default_gate(&mut filter, 3.0);
            assert_eq!(status, AegisStatusCode::Ok);

            // Passing 0.0 falls back to configured 3.0 default gate -> rejects spike
            let status = aegis_scalar_filter_update(&mut filter, 9999.0, 0.0, &mut out);
            assert_eq!(status, AegisStatusCode::MeasurementGated);

            // Passing explicit AEGIS_GATE_DISABLE (-1.0) allows through
            let status =
                aegis_scalar_filter_update(&mut filter, 1.05, AEGIS_GATE_DISABLE, &mut out);
            assert_eq!(status, AegisStatusCode::Ok);
        }
    }

    #[test]
    fn h_infinity_filter() {
        use aegis_filter::ffi::{
            aegis_h_inf_filter_get_bound, aegis_h_inf_filter_init, aegis_h_inf_filter_predict,
            aegis_h_inf_filter_update, AegisHInfFilter,
        };

        unsafe {
            let mut hinf = MaybeUninit::<AegisHInfFilter>::uninit();
            let status = aegis_h_inf_filter_init(hinf.as_mut_ptr(), 0.0, 1.0, 0.01, 0.5, 5.0);
            assert_eq!(status, AegisStatusCode::Ok);
            let mut hinf = hinf.assume_init();

            let status = aegis_h_inf_filter_predict(&mut hinf, 0.1);
            assert_eq!(status, AegisStatusCode::Ok);

            let mut out = 0.0;
            let status = aegis_h_inf_filter_update(&mut hinf, 1.05, &mut out);
            assert_eq!(status, AegisStatusCode::Ok);
            assert!(out > 0.0);
            assert!(aegis_h_inf_filter_get_bound(&hinf) > 0.0);
        }
    }

    #[test]
    fn fusion4_and_fdir() {
        use aegis_filter::ffi::{
            aegis_fusion4_fuse, aegis_fusion4_get_sensor_health, aegis_fusion4_get_state,
            aegis_fusion4_init, aegis_fusion4_predict, aegis_fusion4_register_sensor, AegisFusion4,
            AegisFusionStatusCode,
        };

        unsafe {
            let mut fusion = MaybeUninit::<AegisFusion4>::uninit();
            let status = aegis_fusion4_init(fusion.as_mut_ptr(), 0.0, 1.0, 0.01);
            assert_eq!(status, AegisStatusCode::Ok);
            let mut fusion = fusion.assume_init();

            // Register Sensor 1
            let status = aegis_fusion4_register_sensor(&mut fusion, 1, 0.2, 2, 3.0, 2);
            assert_eq!(status, AegisStatusCode::Ok);

            let status = aegis_fusion4_predict(&mut fusion, 0.5);
            assert_eq!(status, AegisStatusCode::Ok);

            let mut fusion_status = AegisFusionStatusCode::UnregisteredSensor;
            let status = aegis_fusion4_fuse(&mut fusion, 1, 0.52, &mut fusion_status);
            assert_eq!(status, AegisStatusCode::Ok);
            assert_eq!(fusion_status, AegisFusionStatusCode::Fused);
            assert!((aegis_fusion4_get_state(&fusion) - 0.52).abs() < 0.2);
            assert_eq!(aegis_fusion4_get_sensor_health(&fusion, 1), 1.0);
        }
    }
}
