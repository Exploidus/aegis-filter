// ======================================================================
// \title  AegisGncFilter.hpp
// \author Exploidus (https://github.com/Exploidus)
// \brief  NASA F Prime C++ Component integrating aegis-filter state estimation
// ======================================================================

#ifndef EXPLOIDUS_AEGIS_GNC_FILTER_HPP
#define EXPLOIDUS_AEGIS_GNC_FILTER_HPP

#include "fprime/AegisGncFilter/AegisGncFilterComponentAc.hpp"
#include "aegis_filter.h"

namespace Exploidus {
namespace Gnc {

  class AegisGncFilter :
    public AegisGncFilterComponentBase
  {

    public:

      // ----------------------------------------------------------------------
      // Construction, initialization, and destruction
      // ----------------------------------------------------------------------

      //! Construct object AegisGncFilter
      //!
      AegisGncFilter(
          const char *const compName /*!< The component name*/
      );

      //! Initialize component
      //!
      void init(
          const NATIVE_INT_TYPE instance = 0 /*!< The instance number*/
      );

      //! Configure flight filter parameters and sensor calibrations
      //!
      void configure(
          double initial_state,
          double initial_variance,
          double process_noise_q
      );

      //! Destroy object AegisGncFilter
      //!
      ~AegisGncFilter();

    PRIVATE:

      // ----------------------------------------------------------------------
      // Handler implementations for user-defined typed input ports
      // ----------------------------------------------------------------------

      //! Handler implementation for schedIn (Rate group trigger, e.g. 100 Hz)
      //!
      void schedIn_handler(
          const NATIVE_INT_TYPE portNum, /*!< The port number*/
          NATIVE_UINT_TYPE context /*!< The call order*/
      );

      //! Handler implementation for sensorIn (Raw sensor measurements)
      //!
      void sensorIn_handler(
          const NATIVE_INT_TYPE portNum, /*!< The port number*/
          Fw::Buffer &sendBuffer
      );

      // ----------------------------------------------------------------------
      // Member variables
      // ----------------------------------------------------------------------

      //! Low-level C-ABI fusion engine instance
      AegisFusion4 m_fusion;

      //! Total counter of blocked adversarial anomaly spikes
      U32 m_gatedCount;

      //! Mutex guard or initialization flag
      bool m_configured;

    };

} // end namespace Gnc
} // end namespace Exploidus

#endif
