# NASA F Prime (F´) Flight Software Integration Guide

This directory contains the reference integration of **`aegis-filter`** into **NASA's F Prime (F´)** flight software framework.

Developed under **Exploidus** for mission-critical aerospace, satellite Guidance, Navigation, and Control (GNC), and autonomous drone avionics.

---

## 1. Architecture Overview

NASA F Prime operates on a component-based, port-driven architecture where flight software modules communicate via typed synchronous and asynchronous ports.

```
                          +---------------------------------------------------+
                          |            Exploidus::Gnc::AegisGncFilter         |
                          |                                                   |
   [RateGroup 100Hz] ---->| schedIn() ----------> aegis_fusion4_predict()     |
                          |                             |                     |
   [Raw Sensor Telemetry]>| sensorIn() ---------> aegis_fusion4_fuse()        |
                          |                             |                     |
                          |       [3-Sigma Gate]        |                     |
                          |       +-- Spoof Detected? ->|---> eventOut        |
                          |       |                     |     (SensorAnomaly) |
                          |       |                     |                     |
                          |       +-- Valid Reading? -->|---> gncStateOut     |
                          |                             |     (Clean State)   |
                          |                             |                     |
                          |                             +---> tlmOut          |
                          |                                   (GncState, Var) |
                          +---------------------------------------------------+
```

### Key Technical Properties
- **Zero Heap Allocations:** Compile-time static buffers ensure zero dynamic memory allocation (`malloc`/`free`), avoiding memory fragmentation during orbital missions.
- **Deterministic Latency:** Scalar Joseph update runs in **~5.07 ns**, UKF 2D in **~29.08 ns**, well within 1 kHz (1,000,000 ns) real-time deadlines.
- **Automated FDIR:** Consecutive adversarial anomalies trigger automated sensor quarantine without flight controller lockups.

---

## 2. Component Structure

| File | Description |
| :--- | :--- |
| [`AegisGncFilter.fpp`](AegisGncFilter/AegisGncFilter.fpp) | NASA F Prime FPP component definition modeling ports, telemetry, and events |
| [`AegisGncFilter.hpp`](AegisGncFilter/AegisGncFilter.hpp) | C++ Flight Component Header inheriting from `AegisGncFilterComponentBase` |
| [`AegisGncFilter.cpp`](AegisGncFilter/AegisGncFilter.cpp) | C++ Flight Logic calling `aegis-filter` C-FFI and dispatching F Prime events |
| [`CMakeLists.txt`](AegisGncFilter/CMakeLists.txt) | F Prime module build configuration linking `libaegis_filter.a` |
| [`include/aegis_filter.h`](../include/aegis_filter.h) | Pure C/C++ FFI header declaring all filter functions and status codes |

---

## 3. How to Build & Integrate

### Step 1: Compile `aegis-filter` Static Library
For your target architecture (host x86_64, Linux ARM64, or bare-metal ARM Cortex-M):

```bash
# For native flight computer (Linux x86_64 or aarch64)
cargo build --release

# For embedded microcontroller targets (ARM Cortex-M4/M7)
cargo build --release --no-default-features --target thumbv7em-none-eabihf
```

This generates `libaegis_filter.a` in `target/release/` or `target/<target-triple>/release/`.

### Step 2: Register Component in F Prime Deployment
In your F Prime deployment's `project.cmake` or topology:

```cmake
# Add AegisGncFilter directory to F Prime build
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/path/to/aegis-filter/fprime/AegisGncFilter")
```

In your topology FPP file (`Topology.fpp`):
```fpp
instance aegisGnc: Exploidus.Gnc.AegisGncFilter base id 0x2000
```

Connect the rate group and sensor ports:
```fpp
connections GncConnections {
    rateGroup100Hz.SchedOut[0] -> aegisGnc.schedIn
    sensorHandler.dataOut -> aegisGnc.sensorIn
    aegisGnc.gncStateOut -> flightController.stateIn
}
```

### Step 3: Build F Prime Deployment
```bash
fprime-util generate
fprime-util build
```

---

## 4. Telemetry Channels & Events

### Telemetry Channels
- `GncState` (ID `0x100`): Fused physical state mean (altitude, velocity, attitude angle).
- `GncVariance` (ID `0x101`): Dynamic error variance representing filter confidence.
- `TotalAnomaliesGated` (ID `0x102`): Counter of spoofed sensor readings blocked.
- `Sensor1Health`, `Sensor2Health` (ID `0x103`, `0x104`): Dynamic sensor health score (`1.0` = healthy, `0.0` = quarantined).

### Telemetry Events
- `SensorAnomalyGated` (`WARNING_LO`): Emitted when a sensor measurement deviates past 3-sigma innovation threshold.
- `SensorQuarantined` (`WARNING_HI`): Emitted when repeated spikes trigger automatic sensor isolation.
- `SensorRecovered` (`ACTIVITY_HI`): Emitted when an isolated sensor proves reliability and is restored.
- `FilterFault` (`FATAL`): Emitted on non-finite input or numerical singularity.
