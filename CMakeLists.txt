cmake_minimum_required(VERSION 2.8)

set( CMAKE_CXX_STANDARD 20 )

project( geolocation_surveyor )
set( OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/opencv_build )
set( Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../torches/libtorch/share/cmake/Torch )

set(CMAKE_CXX_STANDARD_LIBRARIES "-lX11 -ldl")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

set( source_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/" )
set( computer_vision "${CMAKE_CURRENT_SOURCE_DIR}/computer_vision/" )
set( class_utils "${CMAKE_CURRENT_SOURCE_DIR}/class_utils/" )
set( config "${CMAKE_CURRENT_SOURCE_DIR}/config/" )

# add boost to project
set( BOOST_ROOT "/usr/include/boost" )
find_package(Boost REQUIRED COMPONENTS system)

# add opencv, torch, and other include directories
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${computer_vision} )
include_directories( ${class_utils} )
include_directories( ${config} )

# add the dlib lib
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/dlib )

# include the plotcpp lib
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/plotcpp )

# add command line parsing tool
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/argparse/include )

add_executable( Source ${source_dir}/Source.cpp ${source_dir}/ImageProcessor.cpp ${source_dir}/ClusterIdentification.cpp)
add_executable( config ${source_dir}/ConfigWindow.cpp ${source_dir}/ImageProcessor.cpp ${source_dir}/ClusterIdentification.cpp)
#target_link_libraries( Source "${TORCH_LIBRARIES}" )
target_link_libraries( Source ${OpenCV_LIBS} )
target_link_libraries( Source dlib::dlib )

target_include_directories( Source PUBLIC ${Boost_INCLUDE_DIRS}) 
target_link_libraries( Source ${Boost_LIBRARIES})
