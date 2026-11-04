/**
 * @file aegis_node.cpp
 * @brief Implementation of high-integrity ROS 2 C++ Node for aegis-filter.
 * @author Exploidus (https://github.com/Exploidus)
 * @copyright 2026 Exploidus. Dual-licensed MIT / Apache-2.0.
 */

#include "aegis_ros2_adapter/aegis_node.hpp"

#include <chrono>

namespace aegis_ros2_adapter
{

AegisFilterNode::AegisFilterNode(const rclcpp::NodeOptions & options)
: Node("aegis_filter_node", options)
{
  initialize_node();
  initialize_filters();

  RCLCPP_INFO(
    this->get_logger(),
    "Aegis-Filter ROS 2 Node initialized. Zero-allocation FDIR pipeline active (Gate: %.1f-sigma).",
    gate_sigma_);
}

void AegisFilterNode::initialize_node()
{
  // Declare parameters with safe fail-closed defaults
  this->declare_parameter<std::string>("frame_id", "odom");
  this->declare_parameter<std::string>("child_frame_id", "base_link");
  this->declare_parameter<double>("initial_state", 0.0);
  this->declare_parameter<double>("initial_variance", 1.0);
  this->declare_parameter<double>("process_noise_q", 0.01);
  this->declare_parameter<double>("measurement_noise_r", 0.1);
  this->declare_parameter<double>("gate_sigma", 3.0);
  this->declare_parameter<int>("max_consecutive_anomalies", 5);
  this->declare_parameter<int>("auto_recovery_threshold", 10);

  // Retrieve parameters
  frame_id_ = this->get_parameter("frame_id").as_string();
  child_frame_id_ = this->get_parameter("child_frame_id").as_string();
  initial_state_ = this->get_parameter("initial_state").as_double();
  initial_variance_ = this->get_parameter("initial_variance").as_double();
  process_noise_q_ = this->get_parameter("process_noise_q").as_double();
  measurement_noise_r_ = this->get_parameter("measurement_noise_r").as_double();
  gate_sigma_ = this->get_parameter("gate_sigma").as_double();
  max_consecutive_anomalies_ = this->get_parameter("max_consecutive_anomalies").as_int();
  auto_recovery_threshold_ = this->get_parameter("auto_recovery_threshold").as_int();

  // Create real-time sensor subscriptions
  rclcpp::QoS qos_sensor = rclcpp::SensorDataQoS();

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    "/imu/data",
    qos_sensor,
    std::bind(&AegisFilterNode::imu_callback, this, std::placeholders::_1));

  pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/pose/sample",
    qos_sensor,
    std::bind(&AegisFilterNode::pose_callback, this, std::placeholders::_1));

  // Create output publishers
  odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
    "/aegis/filtered_odometry",
    rclcpp::SystemDefaultsQoS());

  health_pub_ = this->create_publisher<std_msgs::msg::Float32>(
    "/aegis/sensor_health",
    rclcpp::SystemDefaultsQoS());
}

void AegisFilterNode::initialize_filters()
{
  // Zero-allocation initialization of 4-channel Multi-Sensor Fusion FDIR engine
  AegisStatusCode rc = aegis_fusion4_init(
    &fusion_engine_,
    static_cast<aegis_real>(initial_state_),
    static_cast<aegis_real>(initial_variance_),
    static_cast<aegis_real>(process_noise_q_));

  if (rc != AEGIS_OK) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize aegis_fusion4 engine! Error code: %d", rc);
    throw std::runtime_error("Aegis-Filter initialization failed");
  }

  // Register primary pose/altimeter sensor with Chi-square innovation gating
  rc = aegis_fusion4_register_sensor(
    &fusion_engine_,
    SENSOR_PRIMARY_POSE,
    static_cast<aegis_real>(measurement_noise_r_),
    static_cast<uint32_t>(max_consecutive_anomalies_),
    static_cast<aegis_real>(gate_sigma_),
    static_cast<uint32_t>(auto_recovery_threshold_));

  if (rc != AEGIS_OK) {
    RCLCPP_FATAL(this->get_logger(), "Failed to register primary sensor! Error code: %d", rc);
    throw std::runtime_error("Sensor registration failed");
  }

  filter_initialized_ = true;
}

void AegisFilterNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  if (!filter_initialized_) {
    return;
  }

  rclcpp::Time current_time = msg->header.stamp;

  // Calculate dynamic delta-t for prediction
  if (last_imu_time_.nanoseconds() > 0) {
    double dt = (current_time - last_imu_time_).seconds();
    if (dt > 0.0 && dt < 1.0) {
      // Ingest vertical/linear acceleration as control input
      double accel_z = msg->linear_acceleration.z;
      aegis_real control_u = static_cast<aegis_real>(accel_z * dt);

      // Perform state propagation via C-FFI
      AegisStatusCode rc = aegis_fusion4_predict(&fusion_engine_, control_u);
      if (rc != AEGIS_OK) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          2000,
          "Aegis filter predict error: %d", rc);
      }
    }
  }

  last_imu_time_ = current_time;
}

void AegisFilterNode::pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
{
  if (!filter_initialized_) {
    return;
  }

  // Extract observation (e.g. altitude or position along measurement axis)
  aegis_real raw_measurement = static_cast<aegis_real>(msg->pose.pose.position.z);
  AegisFusionStatusCode fusion_status = AEGIS_FUSION_FUSED;

  // Execute zero-copy Joseph-stabilized measurement update with FDIR gating
  AegisStatusCode rc = aegis_fusion4_fuse(
    &fusion_engine_,
    SENSOR_PRIMARY_POSE,
    raw_measurement,
    &fusion_status);

  if (rc != AEGIS_OK) {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      2000,
      "Aegis fusion FFI call error: %d", rc);
    return;
  }

  // Handle FDIR outcomes
  switch (fusion_status) {
    case AEGIS_FUSION_FUSED:
      fused_count_++;
      break;

    case AEGIS_FUSION_GATED:
      gated_count_++;
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "[AEGIS-FDIR GATE] Outlier measurement rejected! Raw value: %.3f (Total gated: %lu)",
        raw_measurement, gated_count_);
      break;

    case AEGIS_FUSION_QUARANTINED:
      quarantine_count_++;
      RCLCPP_ERROR(
        this->get_logger(),
        "[AEGIS-FDIR QUARANTINE] Sensor %u breached anomaly threshold! Quarantined from state.",
        SENSOR_PRIMARY_POSE);
      break;

    case AEGIS_FUSION_IGNORED_QUARANTINED:
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        5000,
        "[AEGIS-FDIR] Ingest ignored for quarantined sensor %u.",
        SENSOR_PRIMARY_POSE);
      break;

    case AEGIS_FUSION_RECOVERED:
      RCLCPP_INFO(
        this->get_logger(),
        "[AEGIS-FDIR RECOVERY] Sensor %u passed consecutive valid sample gate. Restored to nominal.",
        SENSOR_PRIMARY_POSE);
      break;

    default:
      break;
  }

  // Read filtered state and covariance
  aegis_real filtered_state = aegis_fusion4_get_state(&fusion_engine_);
  aegis_real filtered_variance = aegis_fusion4_get_variance(&fusion_engine_);
  aegis_real sensor_health = aegis_fusion4_get_sensor_health(&fusion_engine_, SENSOR_PRIMARY_POSE);

  // Construct and publish filtered Odometry message
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = msg->header.stamp;
  odom_msg.header.frame_id = frame_id_;
  odom_msg.child_frame_id = child_frame_id_;

  // Set position (preserve x and y from pose, replace z with high-integrity estimate)
  odom_msg.pose.pose.position.x = msg->pose.pose.position.x;
  odom_msg.pose.pose.position.y = msg->pose.pose.position.y;
  odom_msg.pose.pose.position.z = static_cast<double>(filtered_state);
  odom_msg.pose.pose.orientation = msg->pose.pose.orientation;

  // Fill diagonal covariance (row-major 6x6 matrix, index 14 is z-variance)
  odom_msg.pose.covariance[14] = static_cast<double>(filtered_variance);

  odom_pub_->publish(odom_msg);

  // Publish real-time sensor health metric
  std_msgs::msg::Float32 health_msg;
  health_msg.data = static_cast<float>(sensor_health);
  health_pub_->publish(health_msg);
}

}  // namespace aegis_ros2_adapter
