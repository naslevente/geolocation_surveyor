#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>

//#include <boost/stacktrace.hpp>
#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>
//#include <torch/torch.h>

#include "ImageProcessor.hpp"
#include "ClusterIdentification.hpp"
#include "Configuration.hpp"

cv::Vec2f getProportionalityConstant(int desiredWidth, int desiredHeight, int currentWidth, int currentHeight) {

    float fx = (float)(desiredWidth) / currentWidth;
    float fy = (float)(desiredHeight) / currentHeight;
    return cv::Vec2f(fx, fy);
}

int main(int argc, char* argv[]) {

    // command line parsing
    argparse::ArgumentParser parser("Geolocation Surveyor");
    parser.add_argument("video").help("<video>.mp4: path to input video");
    parser.add_argument("config").help("<config>.txt: path to config file");

    // make sure necessary input paths are given
    try {

        parser.parse_args(argc, argv);

    } catch(const std::runtime_error &err) {

        std::cerr << err.what() << '\n';
        std::cerr << parser;
        std::exit(1);
    }

    // get input paths from parser
    auto pathToVid = parser.get<std::string>("video");
    auto pathToConfig = parser.get<std::string>("config");

    std::cout << "path to video: " << pathToVid << '\n';
    std::cout << "path to config file: " << pathToConfig << '\n';

    // load video from input location
    cv::VideoCapture inputData ;
    inputData.open(pathToVid);
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
    for(int i = 0; i < 100; ++i) {

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
