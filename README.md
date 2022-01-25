# geolocation_surveyor
## To Clone
`git clone https://github.com/naslevente/geolocation_surveyor.git`

## Requirements
- Working C++ compiler (C++ 17 or greater)
- CMake (version >= 2.8)

## To build
- create two new directories in the directory which will hold the clone of this project: `mkdir torches` and `mkdir opencv`
- cd into opencv and execute the following commands: `wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip` and `unzip opencv.zip`
- create a opencv_build directory: `mkdir opencv_build`
- cd into it and run the following cmake commands: `cmake ../opencv-master` and `cmake --build .`
- enter into the torches directory
- download the stable, linux, libtorch, cpu version of pytorch from https://pytorch.org/
- move the downloaded zip into the torches directory and unzip
- cd into project directory
- clone dlib library into project directory from https://github.com/davisking/dlib
- clone plotcpp wrapper into project directory from https://github.com/Kolkir/plotcpp
- cd back to the directory where the clone of this project will reside
- Create a build directory !IMPORTANT! Use out-of-tree build directory!  (e.g. `mkdir build`)
- cd into build directory (e.g. `cd build`)
- generate CMake configuration `cmake /path/to/repo/geolocation_surveyor`
- build all examples `cmake --build .`
