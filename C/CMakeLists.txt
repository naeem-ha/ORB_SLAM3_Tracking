cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM3_Tracking)

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Include ORB-SLAM3 headers
include_directories(include)

# Check if source files exist
file(GLOB SOURCES src/main.cc src/ORBextractor.cc)
message(STATUS "Sources found: ${SOURCES}")

# Source files
add_executable(orb_extract src/main.cc src/ORBextractor.cc)

# Link libraries
target_link_libraries(orb_extract ${OpenCV_LIBS})