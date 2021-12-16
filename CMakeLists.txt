cmake_minimum_required(VERSION 2.8)

#set_property( CMAKE_CXX_STANDARD 17 )

project( geolocation_surveyor )
set( OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/opencv_build )
set( Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../torches/libtorch/share/cmake/Torch )

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

set( source_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/" )
set( computer_vision "${CMAKE_CURRENT_SOURCE_DIR}/computer_vision/" )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${computer_vision} )

add_executable( Source ${source_dir}/Source.cpp )
target_link_libraries( Source "${TORCH_LIBRARIES}" )
target_link_libraries( Source ${OpenCV_LIBS} )