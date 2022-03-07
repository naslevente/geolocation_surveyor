#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>

//#include <boost/stacktrace.hpp>
#include <opencv2/opencv.hpp>
//#include <torch/torch.h>

#include "ImageProcessor.hpp"
#include "ClusterIdentification.hpp"

cv::Vec2f getProportionalityConstant(int desiredWidth, int desiredHeight, int currentWidth, int currentHeight) {

    float fx = (float)(desiredWidth) / currentWidth;
    float fy = (float)(desiredHeight) / currentHeight;
    return cv::Vec2f(fx, fy);
}

int main(int argv, char* argc[]) {

    if(argv < 2) {

        std::cerr << "error: missing file input location" << '\n';
        exit(-1);
    }

    // load video from input location
    cv::VideoCapture inputData ;
    inputData.open(argc[1]);
    if(!inputData.isOpened()) {

        std::cerr << "error: failed to open video" << '\n';
        exit(-1);
    }

    // create ImageProcessor object
    std::unique_ptr<ImageProcessor> imageProcessor = std::make_unique<ImageProcessor>();
    if(imageProcessor == nullptr) {

        std::cerr << "failed to create ImageProcessor object" << '\n';
        exit(-1);
    }

    // go through each frame from video
    cv::Mat prevFrame;
    cv::Mat currentFrame;
    cv::Mat opticalFlowOutput;

    // skip first 50 frames
    for(int i = 0; i < 60; ++i) {

        inputData >> prevFrame;
    }
    for(int i = 0; i < 10; ++i) {

        inputData >> currentFrame;
    }

    // resize frames to appropriate dimensions
    cv::Vec2f propConsts = std::move(getProportionalityConstant(720, 420, prevFrame.cols, prevFrame.rows));
    cv::resize(prevFrame, prevFrame, cv::Size(), propConsts[0], propConsts[1]);
    cv::resize(currentFrame, currentFrame, cv::Size(), propConsts[0], propConsts[1]);

    std::cout << "dimensions of resized frames: " << prevFrame.cols << " " << prevFrame.rows << '\n';

    // find optical flow between two frames
    imageProcessor->ShowImage(prevFrame);
    imageProcessor->ShowImage(currentFrame);
    imageProcessor->OpticalFlowCalculation(prevFrame, currentFrame, opticalFlowOutput);

    /*
    for( ; ; ) {

        inputData >> currentFrame;
        cv::imshow("current frame", currentFrame);
        cv::waitKey();

        if(!prevFrame.empty()) {

            cv::imshow("previous frame", prevFrame);
            cv::waitKey();
        }

        prevFrame = std::move(currentFrame);

        char c = (char) cv::waitKey(25);
        if(c == 27) break;
    }

    // clean video opener
    inputData.release();
    cv::destroyAllWindows();
    */
}
