// ======================================================================
// \title  AegisGncFilter.cpp
// \author Exploidus (https://github.com/Exploidus)
// \brief  NASA F Prime C++ Component implementation for aegis-filter
// ======================================================================

#include "fprime/AegisGncFilter/AegisGncFilter.hpp"
#include "Fw/Types/BasicTypes.hpp"
#include <cstring>

namespace Exploidus {
namespace Gnc {

  // ----------------------------------------------------------------------
  // Construction, initialization, and destruction
  // ----------------------------------------------------------------------

  AegisGncFilter ::
    AegisGncFilter(
        const char *const compName
    ) :
      AegisGncFilterComponentBase(compName),
      m_gatedCount(0),
      m_configured(false)
  {
      std::memset(&this->m_fusion, 0, sizeof(this->m_fusion));
  }

  void AegisGncFilter ::
    init(
        const NATIVE_INT_TYPE instance
    )
  {
    AegisGncFilterComponentBase::init(instance);
  }

  void AegisGncFilter ::
    configure(
        double initial_state,
        double initial_variance,
        double process_noise_q
    )
  {
    AegisStatusCode status = aegis_fusion4_init(
        &this->m_fusion,
        initial_state,
        initial_variance,
        process_noise_q
    );

    if (status != AEGIS_OK) {
        this->log_FATAL_FilterFault(static_cast<I32>(status));
        return;
    }

    // Default Configuration:
    // Sensor 1: Primary Barometer / Star Tracker (R=0.2, max 3 consecutive anomalies, 3.0-sigma gating)
    aegis_fusion4_register_sensor(&this->m_fusion, 1, 0.20, 3, 3.0, 2);

    // Sensor 2: Secondary GPS / Radio Navigation (R=1.5, max 2 consecutive anomalies, 3.0-sigma gating)
    aegis_fusion4_register_sensor(&this->m_fusion, 2, 1.50, 2, 3.0, 2);

    this->m_configured = true;
  }

  AegisGncFilter ::
    ~AegisGncFilter()
  {

  }

  // ----------------------------------------------------------------------
  // Handler implementations for user-defined typed input ports
  // ----------------------------------------------------------------------

  void AegisGncFilter ::
    schedIn_handler(
        const NATIVE_INT_TYPE portNum,
        NATIVE_UINT_TYPE context
    )
  {
    if (!this->m_configured) {
        return;
    }

    // Predict step: Propagate GNC state forward with time increment dt
    // In actual flight software, control_input can be derived from accelerometer / thruster command
    AegisStatusCode status = aegis_fusion4_predict(&this->m_fusion, 0.0);
    if (status != AEGIS_OK) {
        this->log_FATAL_FilterFault(static_cast<I32>(status));
        return;
    }

    // Emit live telemetry to NASA F Prime downlink
    double currentState = aegis_fusion4_get_state(&this->m_fusion);
    double currentVar = aegis_fusion4_get_variance(&this->m_fusion);
    double health1 = aegis_fusion4_get_sensor_health(&this->m_fusion, 1);
    double health2 = aegis_fusion4_get_sensor_health(&this->m_fusion, 2);

    this->tlmWrite_GncState(currentState);
    this->tlmWrite_GncVariance(currentVar);
    this->tlmWrite_TotalAnomaliesGated(this->m_gatedCount);
    this->tlmWrite_Sensor1Health(health1);
    this->tlmWrite_Sensor2Health(health2);
  }

  void AegisGncFilter ::
    sensorIn_handler(
        const NATIVE_INT_TYPE portNum,
        Fw::Buffer &sendBuffer
    )
  {
    if (!this->m_configured || sendBuffer.getSize() < (sizeof(U8) + sizeof(F64))) {
        return;
    }

    // Unpack sensor packet: [U8 sensorId][F64 rawValue]
    U8* data = sendBuffer.getData();
    U8 sensorId = data[0];
    F64 rawValue = 0.0;
    std::memcpy(&rawValue, &data[1], sizeof(F64));

    AegisFusionStatusCode fusionStatus = AEGIS_FUSION_UNREGISTERED;
    AegisStatusCode code = aegis_fusion4_fuse(
        &this->m_fusion,
        sensorId,
        rawValue,
        &fusionStatus
    );

    if (code != AEGIS_OK) {
        this->log_FATAL_FilterFault(static_cast<I32>(code));
        return;
    }

    // Automated FDIR Fault Detection and Isolation Handlers
    switch (fusionStatus) {
        case AEGIS_FUSION_FUSED:
            // Nominal fusion update
            break;

        case AEGIS_FUSION_GATED:
            // Chi-square outlier rejected (spoofing / adversarial glitch)
            this->m_gatedCount++;
            this->log_WARNING_LO_SensorAnomalyGated(sensorId, rawValue);
            break;

        case AEGIS_FUSION_QUARANTINED:
            // Multi-anomaly threshold reached: isolate sensor
            this->m_gatedCount++;
            this->log_WARNING_HI_SensorQuarantined(sensorId);
            break;

        case AEGIS_FUSION_RECOVERED:
            // Auto-recovery condition met: sensor unquarantined
            this->log_ACTIVITY_HI_SensorRecovered(sensorId);
            break;

        case AEGIS_FUSION_IGNORED_QUARANTINED:
        case AEGIS_FUSION_UNREGISTERED:
        default:
            break;
    }
  }

} // end namespace Gnc
} // end namespace Exploidus
