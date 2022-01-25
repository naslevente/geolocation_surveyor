#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>

#include <boost/stacktrace.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "ImageProcessor.hpp"

int main(int argv, char* argc[]) {

    if(argv < 2) {

        std::cerr << "error: missing file input location" << '\n';
        exit(-1);
    }

    // create ImageProcessor object
    cv::Mat inputMat = cv::imread(argc[1], 1);
    cv::resize(inputMat, inputMat, cv::Size(), 0.25, 0.25);
    std::unique_ptr<ImageProcessor> imageProcessor = std::make_unique<ImageProcessor>(inputMat);

    // apply edge detection on input image
    imageProcessor->LocateObstacle();
    //imageProcessor->CornerDetection();

    // print out call stack
    //std::cout << boost::stacktrace::stacktrace() << '\n';
}