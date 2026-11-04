cmake_minimum_required(VERSION 3.8)
project(aegis_ros2_adapter)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
endif()

# Find ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Path to aegis-filter C-FFI header and compiled library
set(AEGIS_FILTER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(AEGIS_FILTER_INCLUDE_DIR "${AEGIS_FILTER_ROOT}/include" CACHE PATH "Path to aegis-filter include directory")
set(AEGIS_FILTER_LIB_DIR "${AEGIS_FILTER_ROOT}/target/release" CACHE PATH "Path to aegis-filter compiled library directory")

# Allow fallback to debug target if release is not yet built
if(NOT EXISTS "${AEGIS_FILTER_LIB_DIR}/libaegis_filter.a" AND NOT EXISTS "${AEGIS_FILTER_LIB_DIR}/libaegis_filter.so")
  if(EXISTS "${AEGIS_FILTER_ROOT}/target/debug/libaegis_filter.a" OR EXISTS "${AEGIS_FILTER_ROOT}/target/debug/libaegis_filter.so")
    set(AEGIS_FILTER_LIB_DIR "${AEGIS_FILTER_ROOT}/target/debug")
  endif()
endif()

find_library(AEGIS_FILTER_LIB
  NAMES aegis_filter
  PATHS ${AEGIS_FILTER_LIB_DIR}
  NO_DEFAULT_PATH
)

if(NOT AEGIS_FILTER_LIB)
  message(WARNING "libaegis_filter library not found at ${AEGIS_FILTER_LIB_DIR}. Run 'cargo build --release --features ffi' in the repository root prior to building.")
  # Define target placeholder so colcon can still parse the package
  add_library(aegis_filter_lib UNKNOWN IMPORTED)
else()
  add_library(aegis_filter_lib UNKNOWN IMPORTED)
  set_target_properties(aegis_filter_lib PROPERTIES
    IMPORTED_LOCATION "${AEGIS_FILTER_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${AEGIS_FILTER_INCLUDE_DIR}"
  )
endif()

# Include directories
include_directories(
  include
  ${AEGIS_FILTER_INCLUDE_DIR}
)

# Core ROS 2 adapter library
add_library(aegis_ros2_adapter_lib
  src/aegis_node.cpp
)

target_include_directories(aegis_ros2_adapter_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${AEGIS_FILTER_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(aegis_ros2_adapter_lib
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_msgs
)

if(TARGET aegis_filter_lib)
  target_link_libraries(aegis_ros2_adapter_lib PUBLIC aegis_filter_lib)
endif()

# Executable node
add_executable(aegis_filter_node
  src/main.cpp
)

target_link_libraries(aegis_filter_node
  aegis_ros2_adapter_lib
)

ament_target_dependencies(aegis_filter_node
  rclcpp
)

# Install targets
install(TARGETS
  aegis_ros2_adapter_lib
  aegis_filter_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
