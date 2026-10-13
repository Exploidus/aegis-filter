//! Multi-sensor fusion engine with dynamic health scoring and Fault Detection, Isolation, and Recovery (FDIR).
//!
//! Designed for mission-critical aerospace, robotics, and maritime telemetry where redundant sensors
//! (e.g., GPS, Barometer, Rangefinder/LiDAR) must be fused safely with real-time adversarial isolation.

use crate::common::{FilterError, FloatOps};

/// Represents an individual sensor source participating in the multi-sensor fusion pipeline.
#[derive(Debug, Clone, Copy)]
pub struct SensorSource {
    /// Sensor unique identifier
    pub id: u8,
    /// Measurement noise variance (R)
    pub variance: f64,
    /// 3-sigma innovation gating threshold
    pub gate_threshold: f64,
    /// Consecutive anomaly count for automated isolation
    pub anomaly_count: u32,
    /// Maximum allowable consecutive anomalies before quarantine
    pub max_anomalies: u32,
    /// Whether this sensor is currently isolated / quarantined
    pub is_quarantined: bool,
    /// Health score from 0.0 (failed) to 1.0 (nominal)
    pub health_score: f64,
}

impl SensorSource {
    /// Creates a new sensor source configuration.
    pub fn new(id: u8, variance: f64, max_anomalies: u32) -> Result<Self, FilterError> {
        if !variance.is_valid_float() || variance <= 0.0 {
            return Err(FilterError::InvalidParameter);
        }
        Ok(Self {
            id,
            variance,
            gate_threshold: 3.0,
            anomaly_count: 0,
            max_anomalies,
            is_quarantined: false,
            health_score: 1.0,
        })
    }

    /// Sets custom chi-square / sigma gating threshold.
    pub fn with_gate_threshold(mut self, threshold: f64) -> Result<Self, FilterError> {
        if !threshold.is_valid_float() || threshold <= 0.0 {
            return Err(FilterError::InvalidParameter);
        }
        self.gate_threshold = threshold;
        Ok(self)
    }

    /// Manually resets quarantine status.
    pub fn reset_quarantine(&mut self) {
        self.is_quarantined = false;
        self.anomaly_count = 0;
        self.health_score = 1.0;
    }
}

/// A measurement packet from an identifiable sensor source.
#[derive(Debug, Clone, Copy)]
pub struct SensorMeasurement {
    /// Sensor ID matching the configured `SensorSource`
    pub sensor_id: u8,
    /// Measured observation value
    pub value: f64,
}

/// Result of a sensor fusion update step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionStatus {
    /// Measurement was successfully fused into the state vector
    Fused,
    /// Measurement violated outlier gate but sensor remains active
    Gated,
    /// Measurement failed and sensor exceeded threshold, triggering immediate quarantine
    Quarantined,
    /// Measurement came from an already quarantined/disabled sensor
    IgnoredQuarantined,
    /// Sensor ID not registered in the fusion engine
    UnregisteredSensor,
}

/// Multi-sensor state estimator with Joseph-stabilized covariance updates and automated FDIR.
#[derive(Debug, Clone)]
pub struct MultiSensorFusion1D<const N: usize> {
    /// Estimated state mean
    pub x: f64,
    /// Estimated state error variance
    pub p: f64,
    /// Process noise variance (Q)
    pub q: f64,
    /// Array of registered sensor sources
    pub sensors: [Option<SensorSource>; N],
}

impl<const N: usize> MultiSensorFusion1D<N> {
    /// Creates a new multi-sensor fusion filter.
    pub fn new(x0: f64, p0: f64, q: f64) -> Result<Self, FilterError> {
        if !x0.is_valid_float() || !p0.is_valid_float() || !q.is_valid_float() {
            return Err(FilterError::NonFiniteInput);
        }
        if p0 <= 0.0 || q < 0.0 {
            return Err(FilterError::InvalidParameter);
        }

        Ok(Self {
            x: x0,
            p: p0,
            q,
            sensors: [None; N],
        })
    }

    /// Registers a sensor source into the fusion engine.
    pub fn register_sensor(&mut self, source: SensorSource) -> Result<(), FilterError> {
        for slot in self.sensors.iter_mut() {
            if let Some(existing) = slot {
                if existing.id == source.id {
                    *slot = Some(source);
                    return Ok(());
                }
            } else {
                *slot = Some(source);
                return Ok(());
            }
        }
        Err(FilterError::CapacityExceeded)
    }

    /// Predicts state forward using linear physical transition.
    pub fn predict(&mut self, control_input: f64) -> Result<(), FilterError> {
        if !control_input.is_valid_float() {
            return Err(FilterError::NonFiniteInput);
        }
        self.x += control_input;
        self.p += self.q;
        Ok(())
    }

    /// Fuses an incoming measurement from a sensor with FDIR checking.
    pub fn fuse(&mut self, measurement: SensorMeasurement) -> Result<FusionStatus, FilterError> {
        if !measurement.value.is_valid_float() {
            return Err(FilterError::NonFiniteInput);
        }

        // Find matching registered sensor
        let mut target_sensor_idx = None;
        for (i, slot) in self.sensors.iter().enumerate() {
            if let Some(ref s) = slot {
                if s.id == measurement.sensor_id {
                    target_sensor_idx = Some(i);
                    break;
                }
            }
        }

        let idx = match target_sensor_idx {
            Some(i) => i,
            None => return Ok(FusionStatus::UnregisteredSensor),
        };

        let sensor = match self.sensors[idx].as_mut() {
            Some(s) => s,
            None => return Ok(FusionStatus::UnregisteredSensor),
        };

        if sensor.is_quarantined {
            return Ok(FusionStatus::IgnoredQuarantined);
        }

        // Calculate innovation and innovation variance
        let innovation = measurement.value - self.x;
        let s = self.p + sensor.variance;
        let sigma_bound = sensor.gate_threshold * s.sqrt_val();

        // Check innovation gate
        if innovation.abs_val() > sigma_bound {
            sensor.anomaly_count += 1;
            sensor.health_score = (sensor.health_score * 0.5).max(0.0);

            if sensor.anomaly_count >= sensor.max_anomalies {
                sensor.is_quarantined = true;
                sensor.health_score = 0.0;
                return Ok(FusionStatus::Quarantined);
            }
            return Ok(FusionStatus::Gated);
        }

        // Measurement passed gate - perform Joseph-stabilized covariance update
        sensor.anomaly_count = sensor.anomaly_count.saturating_sub(1);
        sensor.health_score = (sensor.health_score + 0.1).min(1.0);

        let k = self.p / s;
        self.x += k * innovation;

        // Joseph Form: P = (1 - K) * P * (1 - K) + K * R * K
        let one_minus_k = 1.0 - k;
        self.p = (one_minus_k * self.p * one_minus_k) + (k * sensor.variance * k);

        Ok(FusionStatus::Fused)
    }

    /// Returns current state estimate.
    pub fn state(&self) -> f64 {
        self.x
    }

    /// Returns current estimation error variance.
    pub fn variance(&self) -> f64 {
        self.p
    }

    /// Returns the active health score of a sensor by ID.
    pub fn sensor_health(&self, sensor_id: u8) -> Option<f64> {
        for s in self.sensors.iter().flatten() {
            if s.id == sensor_id {
                return Some(s.health_score);
            }
        }
        None
    }
}
