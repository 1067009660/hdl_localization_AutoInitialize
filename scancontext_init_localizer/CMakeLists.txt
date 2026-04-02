cmake_minimum_required(VERSION 3.0.2)
project(scancontext_init_localizer)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  message_filters
  roscpp
  nav_msgs
  sensor_msgs
  geometry_msgs
  std_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  message_generation
  ndt_omp
)

find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

add_service_files(
  FILES
  EstimateInitialPose.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES scancontext_core
  CATKIN_DEPENDS message_filters roscpp nav_msgs sensor_msgs geometry_msgs tf2 tf2_ros pcl_ros pcl_conversions message_runtime ndt_omp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(scancontext_core
  src/scan_context.cpp
)

target_link_libraries(scancontext_core
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(sc_map_builder_node
  src/sc_map_builder_node.cpp
)
add_dependencies(sc_map_builder_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(sc_map_builder_node
  scancontext_core
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(sc_ndt_initializer_node
  src/sc_ndt_initializer_node.cpp
)
add_dependencies(sc_ndt_initializer_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(sc_ndt_initializer_node
  scancontext_core
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ndt_omp
  ${Boost_LIBRARIES}
)

install(TARGETS
  scancontext_core
  sc_map_builder_node
  sc_ndt_initializer_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
