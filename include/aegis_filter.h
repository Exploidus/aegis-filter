/**
 * @file aegis_filter.h
 * @brief High-Integrity, Cyber-Resilient State Estimation & Robust Filtering Engine C API.
 * @author Exploidus (https://github.com/Exploidus)
 * @copyright 2026 Exploidus. Dual-licensed MIT / Apache-2.0.
 *
 * Designed for NASA F Prime (F´), flight avionics, and mission-critical embedded systems.
 * Provides zero-allocation, deterministic execution for ARM Cortex-M, RISC-V, and x86/x86_64.
 */

#ifndef AEGIS_FILTER_H
#define AEGIS_FILTER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Return codes for aegis-filter C-FFI operations.
 */
typedef enum {
    AEGIS_OK = 0,                       /**< Operation completed successfully */
    AEGIS_ERR_NON_FINITE_INPUT = -1,    /**< Input contains NaN or Inf */
    AEGIS_ERR_INVALID_COVARIANCE = -2,  /**< Covariance is non-positive or ill-conditioned */
    AEGIS_ERR_SINGULAR_INNOVATION = -3, /**< Innovation covariance is singular */
    AEGIS_ERR_MEASUREMENT_GATED = -4,   /**< Measurement rejected by Chi-square gating */
    AEGIS_ERR_NUMERICAL_INSTABILITY = -5,/**< Riccati bound violated (H-infinity infeasible) */
    AEGIS_ERR_INVALID_PARAMETER = -6,   /**< Parameter out of physical domain */
    AEGIS_ERR_CAPACITY_EXCEEDED = -7,   /**< Sensor registry capacity exceeded */
    AEGIS_ERR_NULL_POINTER = -8         /**< Null pointer passed as argument */
} AegisStatusCode;

/**
 * @brief Status outcome of an individual multi-sensor fusion update.
 */
typedef enum {
    AEGIS_FUSION_FUSED = 0,             /**< Measurement accepted and fused */
    AEGIS_FUSION_GATED = 1,             /**< Gated out as an adversarial outlier */
    AEGIS_FUSION_QUARANTINED = 2,       /**< Sensor exceeded anomaly threshold and is quarantined */
    AEGIS_FUSION_IGNORED_QUARANTINED = 3,/**< Sensor is actively quarantined; reading discarded */
    AEGIS_FUSION_RECOVERED = 4,         /**< Quarantined sensor recovered after consecutive valid samples */
    AEGIS_FUSION_UNREGISTERED = 5       /**< Unknown sensor ID */
} AegisFusionStatusCode;

/* Forward-declared opaque handles */
typedef struct AegisScalarFilter AegisScalarFilter;
typedef struct AegisHInfFilter AegisHInfFilter;
typedef struct AegisFusion4 AegisFusion4;

/* ========================================================================= */
/* 1. SCALAR KALMAN FILTER (1D Joseph Form + Outlier Gating)                 */
/* ========================================================================= */

/**
 * @brief Initializes a 1D Scalar Kalman Filter in pre-allocated memory.
 *
 * @param filter Pointer to filter struct.
 * @param initial_state Initial state mean (x0).
 * @param initial_variance Initial error covariance (P0 > 0).
 * @param process_noise_q Process disturbance variance (Q >= 0).
 * @param measurement_noise_r Measurement noise variance (R > 0).
 * @return AEGIS_OK on success, or error code.
 */
AegisStatusCode aegis_scalar_filter_init(
    AegisScalarFilter* filter,
    double initial_state,
    double initial_variance,
    double process_noise_q,
    double measurement_noise_r
);

/**
 * @brief Propagates filter state forward in time: x = x + u, P = P + Q.
 */
AegisStatusCode aegis_scalar_filter_predict(
    AegisScalarFilter* filter,
    double control_input
);

/**
 * @brief Updates filter state using Joseph-stabilized covariance formulation.
 *
 * @param filter Pointer to filter.
 * @param measurement Raw incoming sensor observation (z).
 * @param gate_threshold_sigma Gating threshold in sigmas (e.g. 3.0, or <= 0 to disable).
 * @param out_state Optional pointer to receive updated state.
 * @return AEGIS_OK on success, AEGIS_ERR_MEASUREMENT_GATED if rejected.
 */
AegisStatusCode aegis_scalar_filter_update(
    AegisScalarFilter* filter,
    double measurement,
    double gate_threshold_sigma,
    double* out_state
);

/**
 * @brief Retrieves current state mean.
 */
double aegis_scalar_filter_get_state(const AegisScalarFilter* filter);

/**
 * @brief Retrieves current error variance.
 */
double aegis_scalar_filter_get_variance(const AegisScalarFilter* filter);

/* ========================================================================= */
/* 2. H-INFINITY MINIMAX ROBUST FILTER (Bounded Disturbance)                  */
/* ========================================================================= */

/**
 * @brief Initializes an H-infinity minimax filter.
 *
 * @param filter Pointer to filter struct.
 * @param initial_state Initial state mean.
 * @param initial_variance Initial error bound.
 * @param process_bound_q Process disturbance bound.
 * @param measurement_bound_r Measurement disturbance bound.
 * @param gamma Performance attenuation bound (gamma > 0, typically >= 1.0).
 */
AegisStatusCode aegis_h_inf_filter_init(
    AegisHInfFilter* filter,
    double initial_state,
    double initial_variance,
    double process_bound_q,
    double measurement_bound_r,
    double gamma
);

/**
 * @brief Propagates H-infinity state forward: x = x + u, P = P + Q.
 */
AegisStatusCode aegis_h_inf_filter_predict(
    AegisHInfFilter* filter,
    double control_input
);

/**
 * @brief Updates state under discrete-time bounded-real Riccati criteria.
 */
AegisStatusCode aegis_h_inf_filter_update(
    AegisHInfFilter* filter,
    double measurement,
    double* out_state
);

/**
 * @brief Retrieves current H-infinity state estimate.
 */
double aegis_h_inf_filter_get_state(const AegisHInfFilter* filter);

/**
 * @brief Retrieves current error bound (P).
 */
double aegis_h_inf_filter_get_bound(const AegisHInfFilter* filter);

/* ========================================================================= */
/* 3. MULTI-SENSOR FDIR FUSION ENGINE (4 Channels)                           */
/* ========================================================================= */

/**
 * @brief Initializes a 4-channel Multi-Sensor Fusion FDIR engine.
 */
AegisStatusCode aegis_fusion4_init(
    AegisFusion4* fusion,
    double x0,
    double p0,
    double q
);

/**
 * @brief Registers an individual sensor into the flight fusion engine.
 *
 * @param fusion Pointer to fusion engine.
 * @param sensor_id Unique sensor ID (e.g. 1=Baro, 2=GPS, 3=LiDAR).
 * @param variance Measurement noise variance (R).
 * @param max_consecutive_anomalies Number of consecutive spikes before quarantine.
 * @param gate_threshold_sigma Gating threshold in sigmas (default: 3.0).
 * @param auto_recovery_threshold Consecutive clean samples needed to unquarantine (0=disabled).
 */
AegisStatusCode aegis_fusion4_register_sensor(
    AegisFusion4* fusion,
    uint8_t sensor_id,
    double variance,
    uint32_t max_consecutive_anomalies,
    double gate_threshold_sigma,
    uint32_t auto_recovery_threshold
);

/**
 * @brief Propagates fusion state forward.
 */
AegisStatusCode aegis_fusion4_predict(
    AegisFusion4* fusion,
    double control_input
);

/**
 * @brief Fuses incoming measurement with automated Fault Detection & Isolation.
 *
 * @param fusion Pointer to fusion engine.
 * @param sensor_id ID of originating sensor.
 * @param value Raw sensor value.
 * @param out_status Pointer to receive fusion decision outcome.
 */
AegisStatusCode aegis_fusion4_fuse(
    AegisFusion4* fusion,
    uint8_t sensor_id,
    double value,
    AegisFusionStatusCode* out_status
);

/**
 * @brief Retrieves fused state estimate.
 */
double aegis_fusion4_get_state(const AegisFusion4* fusion);

/**
 * @brief Retrieves fused error covariance.
 */
double aegis_fusion4_get_variance(const AegisFusion4* fusion);

/**
 * @brief Retrieves real-time health score of a sensor (1.0 = nominal, 0.0 = quarantined).
 */
double aegis_fusion4_get_sensor_health(const AegisFusion4* fusion, uint8_t sensor_id);

#ifdef __cplusplus
}
#endif

#endif /* AEGIS_FILTER_H */
