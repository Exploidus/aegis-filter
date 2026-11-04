/**
 * @file main.cpp
 * @brief Executable entrypoint for the aegis_ros2_adapter node.
 * @author Exploidus (https://github.com/Exploidus)
 * @copyright 2026 Exploidus. Dual-licensed MIT / Apache-2.0.
 */

#include "aegis_ros2_adapter/aegis_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<aegis_ros2_adapter::AegisFilterNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("aegis_main"), "Exception caught: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
