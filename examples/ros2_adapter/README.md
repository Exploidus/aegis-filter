# ROS 2 C++ Node Adapter for aegis-filter

High-integrity, zero-allocation Robot Operating System 2 (ROS 2) C++ node adapter integrating the **`aegis-filter`** state estimation, sensor fusion, and Fault Detection, Isolation, and Recovery (FDIR) engine into modern robotics stacks.

Designed for Autonomous Mobile Robots (AMRs), industrial manipulators, UAVs, and autonomous driving architectures where **adversarial outlier rejection**, **numerical stability**, and **deterministic execution** are mission-critical.

---

## 1. Architectural Overview

```
                                ROS 2 Topics
                                     │
      /imu/data                      │      /pose/sample
 (sensor_msgs/msg/Imu)               │  (geometry_msgs/msg/PoseWithCovarianceStamped)
           │                         │                  │
           ▼                         │                  ▼
┌─────────────────────────┐          │      ┌─────────────────────────┐
│   imu_callback(msg)     │          │      │   pose_callback(msg)    │
│  - Compute dt           │          │      │  - Extract raw z-pos    │
│  - Control u = a_z * dt │          │      │  - FDIR Chi-square gate │
└────────────┬────────────┘          │      └───────────┬─────────────┘
             │                       │                  │
             ▼                       │                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    aegis-filter C-FFI Engine                        │
│                                                                     │
│  • aegis_fusion4_predict()  ──> Propagate state & covariance        │
│  • aegis_fusion4_fuse()     ──> Joseph-form covariance update       │
│  • Chi-square Gating (3.0σ) ──> Discard adversarial / spoofed spikes│
│  • Health Degradation (FDIR)──> Quarantine persistent anomalies     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │            Published Topics               │
             │  • /aegis/filtered_odometry (nav_msgs)    │
             │  • /aegis/sensor_health (std_msgs)        │
             └───────────────────────────────────────────┘
```

### Key Performance & Safety Guarantees
- **Zero Heap Allocations on Hot Path**: The `AegisFusion4` engine struct is embedded directly as a value member within the `AegisFilterNode` class, eliminating runtime `malloc`/`free` calls during high-rate callbacks.
- **Joseph Form Covariance Updates**: Unconditionally guarantees positive semi-definiteness and matrix symmetry under finite floating-point arithmetic.
- **Automated FDIR Gating**: Transient sensor glitches or adversarial spoofing bursts are rejected before state contamination, with automated recovery after consecutive valid samples.

---

## 2. Prerequisites

- **ROS 2**: Compatible with ROS 2 **Humble Hawksbill**, **Iron Irwini**, **Jazzy Jalisco**, or **Rolling Ridley**.
- **Rust Toolchain**: Rust stable (`1.70+`) with `cargo`.
- **C++ Compiler**: GCC 9+ or Clang 11+ with C++17 support.
- **Build Tools**: `colcon` and `ament_cmake`.

---

## 3. Build Instructions (Cargo + Colcon)

Building the adapter is a standard two-stage process: compile the Rust C-FFI static library, then build the ROS 2 package using `colcon`.

### Step 1: Build the aegis-filter C-FFI Library

From the repository root:

```bash
# Build the release static library (produces target/release/libaegis_filter.a)
cargo build --release --features ffi
```

For ARM Cortex-M or single-precision robotics platforms:
```bash
cargo build --release --no-default-features --features ffi,single-precision
```

### Step 2: Build the ROS 2 Package with Colcon

From your ROS 2 workspace root (e.g. `~/ros2_ws`):

```bash
# Source your ROS 2 distribution
source /opt/ros/humble/setup.bash

# Option A: Symlink or place this repository in your workspace src/ directory:
# ln -s /path/to/aegis-filter/examples/ros2_adapter src/aegis_ros2_adapter

# Option B: Build directly pointing to this package directory:
colcon build --packages-select aegis_ros2_adapter --symlink-install
```

### Step 3: Source the Workspace Overlay

```bash
source install/setup.bash
```

---

## 4. Running the Node

Run the node with default parameters:

```bash
ros2 run aegis_ros2_adapter aegis_filter_node
```

### Node Parameters

Parameters can be passed via command line or a YAML configuration file:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `frame_id` | `string` | `"odom"` | Reference frame for published odometry |
| `child_frame_id` | `string` | `"base_link"` | Child frame ID |
| `initial_state` | `double` | `0.0` | Initial state mean ($x_0$) |
| `initial_variance` | `double` | `1.0` | Initial error variance ($P_0 > 0$) |
| `process_noise_q` | `double` | `0.01` | Process noise variance ($Q$) |
| `measurement_noise_r` | `double` | `0.1` | Sensor observation noise variance ($R$) |
| `gate_sigma` | `double` | `3.0` | Chi-square gating threshold ($3.0\sigma$) |
| `max_consecutive_anomalies`| `int` | `5` | Anomalies before quarantine |
| `auto_recovery_threshold` | `int` | `10` | Valid samples required to restore sensor |

Example running with custom 4.0-sigma gating:

```bash
ros2 run aegis_ros2_adapter aegis_filter_node --ros-args -p gate_sigma:=4.0 -p measurement_noise_r:=0.05
```

---

## 5. Verification & Synthetic Attack Testing

### Check Subscribed and Published Topics

```bash
ros2 topic list
```
Expected output:
```text
/aegis/filtered_odometry
/aegis/sensor_health
/imu/data
/pose/sample
/parameter_events
/rosout
```

### Test 1: Ingest Nominal Sensor Data

Publish valid pose measurements:
```bash
ros2 topic pub /pose/sample geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'odom'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 1.05}}}}" -1
```

Monitor filtered odometry:
```bash
ros2 topic echo /aegis/filtered_odometry
```

### Test 2: Ingest Adversarial Outlier (GPS / LiDAR Spoofing)

Publish a wild sensor spike ($z = 999.0$ meters):
```bash
ros2 topic pub /pose/sample geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'odom'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 999.0}}}}" -1
```

**Node Console Log**:
```text
[WARN] [aegis_filter_node]: [AEGIS-FDIR GATE] Outlier measurement rejected! Raw value: 999.000 (Total gated: 1)
```

The filter **rejects** the measurement without contaminating the state, preserving nominal control continuity.

---

## 6. License

Dual-licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))
