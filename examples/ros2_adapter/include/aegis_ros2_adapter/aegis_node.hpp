/**
 * @file aegis_node.hpp
 * @brief High-Integrity ROS 2 C++ Node integrating aegis-filter state estimation.
 * @author Exploidus (https://github.com/Exploidus)
 * @copyright 2026 Exploidus. Dual-licensed MIT / Apache-2.0.
 *
 * Implements a zero-allocation, deterministic ROS 2 sensor fusion adapter node.
 * Binds directly to aegis-filter's C-FFI entrypoints without dynamic heap
 * allocation in the real-time callback loop.
 */

#ifndef AEGIS_ROS2_ADAPTER__AEGIS_NODE_HPP_
#define AEGIS_ROS2_ADAPTER__AEGIS_NODE_HPP_

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32.hpp"

#include "aegis_filter.h"

namespace aegis_ros2_adapter
{

/**
 * @brief Sensor IDs registered in the multi-sensor FDIR fusion engine.
 */
enum SensorChannel : uint8_t
{
  SENSOR_PRIMARY_POSE = 1,
  SENSOR_SECONDARY_ALTIMETER = 2,
};

/**
 * @class AegisFilterNode
 * @brief ROS 2 Node subscribing to IMU and Pose topics, running aegis-filter FDIR,
 *        and publishing filtered odometry and health metrics.
 */
class AegisFilterNode : public rclcpp::Node
{
public:
  /**
   * @brief Construct a new AegisFilterNode with configurable parameters.
   * @param options ROS 2 NodeOptions for composable execution.
   */
  explicit AegisFilterNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /**
   * @brief Destructor ensuring safe lifecycle teardown.
   */
  virtual ~AegisFilterNode() = default;

  /**
   * @brief Get count of successfully fused measurements.
   */
  uint64_t get_fused_count() const { return fused_count_; }

  /**
   * @brief Get count of gated (rejected outlier) measurements.
   */
  uint64_t get_gated_count() const { return gated_count_; }

  /**
   * @brief Get count of quarantined sensor events.
   */
  uint64_t get_quarantine_count() const { return quarantine_count_; }

private:
  /**
   * @brief Initialize parameters, publishers, and subscriptions.
   */
  void initialize_node();

  /**
   * @brief Initialize filter structs with deterministic, pre-allocated memory.
   */
  void initialize_filters();

  /**
   * @brief Callback invoked when raw IMU data arrives (high rate, e.g. 100-500 Hz).
   *        Performs the state propagation (predict) step.
   * @param msg ConstSharedPtr to sensor_msgs::msg::Imu
   */
  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);

  /**
   * @brief Callback invoked when pose / localization data arrives (e.g. 10-50 Hz).
   *        Performs Chi-square gated update and publishes filtered odometry.
   * @param msg ConstSharedPtr to geometry_msgs::msg::PoseWithCovarianceStamped
   */
  void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);

  // Subscriptions & Publishers
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr health_pub_;

  // Pre-allocated filter memory (Zero heap allocation in real-time execution path)
  AegisFusion4 fusion_engine_;

  // Node parameters
  std::string frame_id_;
  std::string child_frame_id_;
  double initial_state_;
  double initial_variance_;
  double process_noise_q_;
  double measurement_noise_r_;
  double gate_sigma_;
  int max_consecutive_anomalies_;
  int auto_recovery_threshold_;

  // Telemetry metrics
  uint64_t fused_count_{0};
  uint64_t gated_count_{0};
  uint64_t quarantine_count_{0};
  rclcpp::Time last_imu_time_{0, 0, RCL_ROS_TIME};
  bool filter_initialized_{false};
};

}  // namespace aegis_ros2_adapter

#endif  // AEGIS_ROS2_ADAPTER__AEGIS_NODE_HPP_
