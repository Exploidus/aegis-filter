module Exploidus {

  module Gnc {

    @ Component for high-integrity, cyber-resilient GNC state estimation and multi-sensor FDIR.
    @ Integrates pure `#![no_std]` aegis-filter with NASA F Prime flight software.
    passive component AegisGncFilter {

      # ----------------------------------------------------------------------
      # Ports
      # ----------------------------------------------------------------------

      @ Rate group schedule port (typically 100 Hz or 1 kHz GNC cycle)
      sync input port schedIn: Svc.Sched

      @ Input port for raw sensor telemetry packets
      sync input port sensorIn: Fw.BufferSend

      @ Output port publishing filtered state & covariance to flight control / actuators
      output port gncStateOut: Fw.BufferSend

      # ----------------------------------------------------------------------
      # Standard F Prime Framework Ports
      # ----------------------------------------------------------------------
      time get port timeCaller
      event port eventOut
      text event port textEventOut
      telemetry port tlmOut

      # ----------------------------------------------------------------------
      # Telemetry Channels
      # ----------------------------------------------------------------------

      @ Filtered state mean estimate
      telemetry GncState: F64 id 0x100

      @ Estimated error covariance (confidence metric)
      telemetry GncVariance: F64 id 0x101

      @ Total count of adversarial spoofing spikes gated
      telemetry TotalAnomaliesGated: U32 id 0x102

      @ Health score for Primary Sensor 1 (1.0 = nominal, 0.0 = quarantined)
      telemetry Sensor1Health: F64 id 0x103

      @ Health score for Secondary Sensor 2 (1.0 = nominal, 0.0 = quarantined)
      telemetry Sensor2Health: F64 id 0x104

      # ----------------------------------------------------------------------
      # Events
      # ----------------------------------------------------------------------

      @ Emitted when an anomalous sensor reading is rejected by innovation gating
      event SensorAnomalyGated(
        sensorId: U8,
        rawValue: F64
      ) severity warning low \
        format "Aegis-Filter GNC: Sensor {} spoof spike gated, value: {}" \
        id 0x10

      @ Emitted when a sensor exceeds consecutive anomaly threshold and is quarantined
      event SensorQuarantined(
        sensorId: U8
      ) severity warning high \
        format "Aegis-Filter GNC: Sensor {} isolated and quarantined by automated FDIR" \
        id 0x11

      @ Emitted when a quarantined sensor passes consecutive healthy checks and recovers
      event SensorRecovered(
        sensorId: U8
      ) severity activity high \
        format "Aegis-Filter GNC: Sensor {} restored to active flight telemetry pool" \
        id 0x12

      @ Critical numerical or covariance failure
      event FilterFault(
        errorCode: I32
      ) severity fatal \
        format "Aegis-Filter GNC: Critical filter failure, code: {}" \
        id 0x13

    }

  }

}
