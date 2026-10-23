####
# NASA F Prime CMake configuration for Exploidus AegisGncFilter Component
####

set(SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/AegisGncFilter.cpp"
)

# Set up F Prime component
register_fprime_module()

# Include paths for aegis-filter C header
target_include_directories(
  ${MODULE_NAME}
  PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}/../../include"
)

# Link the static bare-metal/cross-compiled aegis-filter Rust library
# Target path can be customized depending on cross-compilation (e.g. thumbv7em-none-eabihf or native Linux)
if(DEFINED CMAKE_TOOLCHAIN_FILE)
  set(AEGIS_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/../../target/${CMAKE_SYSTEM_PROCESSOR}-unknown-linux-gnu/release")
else()
  set(AEGIS_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/../../target/release")
endif()

find_library(
  AEGIS_FILTER_LIB
  NAMES aegis_filter libaegis_filter.a
  PATHS ${AEGIS_LIB_DIR}
  NO_DEFAULT_PATH
)

if(AEGIS_FILTER_LIB)
  target_link_libraries(
    ${MODULE_NAME}
    PUBLIC
    ${AEGIS_FILTER_LIB}
  )
else()
  message(STATUS "Aegis-Filter static library not found at ${AEGIS_LIB_DIR}. Run 'cargo build --release' first.")
endif()
