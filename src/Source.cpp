#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>

#include <boost/stacktrace.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "ImageProcessor.hpp"
#include "ClusterIdentification.hpp"

int main(int argv, char* argc[]) {

    if(argv < 2) {

        std::cerr << "error: missing file input location" << '\n';
        exit(-1);
    }

    // load video from input location
    cv::VideoCapture inputData (argc[1]);
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

    inputData >> prevFrame;
    inputData >> currentFrame;

    // find optical flow between two frames
    imageProcessor->ShowImage(prevFrame);
    imageProcessor->ShowImage(currentFrame);
    imageProcessor->OpticalFlowCalculation(prevFrame, currentFrame, opticalFlowOutput);

    /*

    cv::resize(prevFrame, prevFrame, cv::Size(), 0.15, 0.15);
    cv::resize(currentFrame, currentFrame, cv::Size(), 0.15, 0.15);

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

    // create ImageProcessor object
    cv::Mat inputMat = cv::imread(argc[1], 1);
    //cv::resize(inputMat, inputMat, cv::Size(), 0.25, 0.25);
    std::unique_ptr<ImageProcessor> imageProcessor = std::make_unique<ImageProcessor>(inputMat);

    // apply edge detection on input image
    imageProcessor->LocateObstacle();
    //imageProcessor->CornerDetection();

    // print out call stack
    //std::cout << boost::stacktrace::stacktrace() << '\n';

    */
}