#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <cassert>
#include <math.h>
#include <numeric>

//#include <boost/stacktrace.hpp>

#include "ImageProcessor.hpp"
#include "ClusterIdentification.hpp"
#include "class_utils.hpp"
#include "WindowData.hpp"
#include "Worker.hpp"

#define PI 3.14159265

ImageProcessor::ImageProcessor(cv::Mat inputMat) {

    this->inputMat = std::move(inputMat);
}

ImageProcessor::ImageProcessor() { }

ImageProcessor::~ImageProcessor() = default;

void ImageProcessor::ShowImage(const cv::Mat &input) const {

    cv::imshow("output image", input);
    cv::waitKey();
}

void ImageProcessor::ShowImage(const cv::Mat &input, const cv::Mat &input2) const {

    cv::imshow("output image", input);
    cv::imshow("output image 2", input2);

    cv::waitKey();
    cv::waitKey();
}

// different ways to clear up image before calculating optical flow
void ImageProcessor::RemoveNoise(cv::Mat &inputMat) const {

    cv::dilate(inputMat, inputMat, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    cv::erode(inputMat, inputMat, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    //cv::blur(inputMat, inputMat, cv::Size(9, 9));
    //cv::medianBlur(inputMat, inputMat, 7);

    // edge detection
    //cv::blur(inputMat, inputMat, cv::Size(3, 3));
    //cv::Canny(inputMat, inputMat, 90, 90 * 3, 3);
}

// 2 fmap inputs: row/col manipulation, transposition
// TODO: generalize for use by corner detection and trajectory angle calculation
template<typename T, typename ...U>
float ImageProcessor::SubmatrixCreation(std::pair<T, T> &inputPair, U ...args) const {

    // unpack fmap and two integer values
    auto params = std::make_tuple(args...);

    // apply crop and transposition lambdas on input gradient pair
    auto newPair = std::get<0>(params)(inputPair, std::get<2>(params), std::get<3>(params));
    newPair = std::get<1>(params)(newPair);
    newPair = std::get<0>(params)(newPair, std::get<4>(params), std::get<5>(params));

    //std::cout << "resulting submatrix: " << cv::format(std::get<1>(newPair), cv::Formatter::FMT_PYTHON) << '\n';
    //std::cout << "submatrix dimensions: " << std::get<0>(newPair).cols << " " << std::get<0>(newPair).rows << '\n';
    // accumulate values in submatrix
    auto accumulation = [](cv::Mat input) -> float {

        int finalSum = 0;
        for(int i = 0; i < input.rows; ++i) {

            const auto row = input.row(i);
            finalSum += std::accumulate(row.begin<float>(), row.end<float>(), 0, 
                [](float a, float b) { return abs(b) + a; });
        }

        return finalSum;
    };

    return (accumulation(std::get<0>(newPair)) + accumulation(std::get<1>(newPair)));
}

cv::Mat ImageProcessor::lFunc(const cv::Mat &input, int param1, int param2) {

    return input.rowRange(param1, param2);
}

cv::Mat ImageProcessor::lTrans(const cv::Mat &input) {

    return input.t();
}

void ImageProcessor::LocateObstacle() {

    // define me constants
    const int maxThreshold = 100;
    const int lowThreshold = 90;
    const int kernel = 3;
    const int ratio = 3;

    // convert rgb mat to grayscale
    cv::Mat grayscale;
    cv::cvtColor(this->inputMat, grayscale, cv::COLOR_BGR2GRAY);

    ShowImage(grayscale);

    // edge detection with canny
    cv::Mat finalImage;
    cv::blur(grayscale, finalImage, cv::Size(3, 3));
    cv::Canny(finalImage, finalImage, lowThreshold, lowThreshold * ratio, kernel);

    ShowImage(finalImage);

    // move grayscale mat into object's gray image field variable
    this->grayImage = finalImage;

    // calculate image gradients and process gradients to find specialized points
    cv::Mat grad_x;
    cv::Mat grad_y;
    this->ImageGradientCalculation(finalImage, grad_x, grad_y);
    //this->ProcessGradients(grad_x, grad_y, cv::Size(9, 9));
}

void ImageProcessor::CornerDetection() const {

    // define constants
    const int blockSize = 2;
    const int apertureSize = 3;
    const double k = 0.04;
    const int threshold = 200;

    ShowImage(this->grayImage);

    // opencv corner detection method use
    cv::Mat finalImage = cv::Mat::zeros(this->grayImage.size(), CV_32FC1);
    cv::cornerHarris(this->grayImage, finalImage, blockSize, apertureSize, k);
    ShowImage(finalImage);

    // scaling and normalization
    cv::Mat finalNormalized;
    cv::Mat finalNormalizedScaled;
    cv::normalize(finalImage, finalNormalized, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    ShowImage(finalNormalized);
    cv::convertScaleAbs(finalNormalized, finalNormalizedScaled);

    // draw corner points calculated onto final image
    for(int i = 0; i < finalNormalized.rows; ++i) {

        for(int j = 0; j < finalNormalized.cols; ++j) {

            if((int)finalNormalized.at<float>(i, j) > threshold) {

                circle(finalNormalizedScaled, cv::Point(j,i), 1,  cv::Scalar(0), 2, 8, 0);
            }
        }
    }

    ShowImage(finalNormalizedScaled);
}

void ImageProcessor::ImageGradientCalculation(cv::Mat inputMat, cv::Mat &grad_x, cv::Mat &grad_y) const {

    // define constants and other necessary variables
    const int kernelSize = 3;
    const int scale = 1; // default value
    const int delta = 0; // default value
    const int depth = CV_16S;

    // calculate gradients with sobel operator
    cv::Sobel(inputMat, grad_x, depth, 1, 0, kernelSize, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(inputMat, grad_y, depth, 0, 1, kernelSize, scale, delta, cv::BORDER_DEFAULT);
}

// calculates optical flow between two input files and outputs optical flow in visual format
void ImageProcessor::OpticalFlowCalculation(cv::Mat &prevFrame, cv::Mat &currentFrame, cv::Mat &outputFrame) const {

    // output frame with black background which will hold all optical flow vectors
    cv::Mat blackOutput { prevFrame.rows, prevFrame.cols, prevFrame.type(), cv::Scalar(0, 0, 0) };

    // change input frames to grayscale (single channeled)
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);
    cv::cvtColor(currentFrame, currentFrame, cv::COLOR_BGR2GRAY);

    // remove noise 
    RemoveNoise(prevFrame);
    RemoveNoise(currentFrame);
    ShowImage(prevFrame);
    ShowImage(currentFrame);

    cv::calcOpticalFlowFarneback(prevFrame, currentFrame, outputFrame, 0.5, 3, 15, 3, 5, 1.2, 0);
    std::cout << "output frame dimensions: " << outputFrame.cols << " " << outputFrame.rows << '\n';
    std::cout << "input frame dimensions: " << prevFrame.cols << " " << prevFrame.rows << '\n';

    // optical flow output values
    //std::cout << cv::format(outputFrame, cv::Formatter::FMT_PYTHON) << '\n';

    // convert prevFrame back to color to see optical flow vector outputs
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_GRAY2BGR);

    // reference vector in top left corner
    cv::arrowedLine(blackOutput, cv::Point(0, 0), cv::Point(2, 6), cv::Scalar(0, 0, 255), 1, 8, 0, 0.1);

    // two separate WindowData objects for two approaches
    WindowData dims1 { cv::Size(-1, 150), 100 };
    WindowData dims2 { cv::Size(20, 150), 100 }; // TODO: generalize height variable of point input

    // iterate through displacement vectors and output onto prevFrame (only every third pixel)
    for(int i = 0; i < outputFrame.rows; i += 9) {

        for(int j = 0; j < outputFrame.cols; j += 9) {

            // create original point, retrieve delta for x and y direction and create final endpoint 
            cv::Point2f originalPoint = cv::Point2f(j, i);
            cv::Vec2f deltas = outputFrame.at<cv::Vec2f>(i, j);
            cv::Point2f endPoint = cv::Point2f((float)(originalPoint.x + deltas[0]), (float)(originalPoint.y + deltas[1]));

            if(i > dims1.getStartRow() && i < dims1.getStartRow() + dims1.getWindowDims().height &&
                j > outputFrame.cols / 3 && j < outputFrame.cols * 2 / 3) {

                std::cout << deltas << " ";

                //cv::arrowedLine(prevFrame, originalPoint, endPoint, cv::Scalar(0, 0, 255), 1, 8, 0, 0.1);
                //cv::arrowedLine(blackOutput, originalPoint, endPoint, cv::Scalar(0, 0, 255), 1, 8, 0, 0.1);

                cv::arrowedLine(prevFrame, originalPoint, endPoint, cv::Scalar(0, 255, 0), 1, 8, 0, 0.1);
                cv::arrowedLine(blackOutput, originalPoint, endPoint, cv::Scalar(0, 255, 0), 1, 8, 0, 0.1);

            }
            else {

                cv::arrowedLine(prevFrame, originalPoint, endPoint, cv::Scalar(0, 255, 0), 1, 8, 0, 0.1);
                cv::arrowedLine(blackOutput, originalPoint, endPoint, cv::Scalar(0, 255, 0), 1, 8, 0, 0.1);
            }
        }
    }

    std::cout << '\n';
    
    cv::line(prevFrame, cv::Point(0, 100), cv::Point(prevFrame.cols, 100), cv::Scalar(0, 0, 0), 2);
    cv::line(prevFrame, cv::Point(0, 250), cv::Point(prevFrame.cols, 250), cv::Scalar(0, 0, 0), 2);
    ShowImage(prevFrame, blackOutput);

    // TODO: remove unnecessary copies
    // output optical flow output magnitudes of small window in center
    cv::Mat croppedOutput = lFunc(outputFrame, dims1.getStartRow(), dims1.getStartRow() + dims1.getWindowDims().height);
    croppedOutput = lTrans(croppedOutput);
    croppedOutput = lFunc(croppedOutput, outputFrame.cols / 3, outputFrame.cols / 3 * 2);
    //std::cout << cv::format(croppedOutput, cv::Formatter::FMT_PYTHON) << '\n';

    // calculated suggested trajectory vector drawing
    int suggestedDirection = FindOptimalDirection(outputFrame, dims1); // Size variable's width must be a multiple of input cols
    int suggestedDirection2 = FindOptimalDirection_(outputFrame, dims2);
    cv::arrowedLine(prevFrame, cv::Point(prevFrame.cols / 2, prevFrame.rows), cv::Point(suggestedDirection, dims1.getStartRow() + (dims1.getWindowDims().height / 2)),
        cv::Scalar(0, 0, 255), 2, 8, 0, 0.04);

    // original motion vector drawing
    cv::arrowedLine(prevFrame, cv::Point(prevFrame.cols / 2, prevFrame.rows), cv::Point(prevFrame.cols / 2, dims1.getStartRow() + (dims1.getWindowDims().height / 2)),
        cv::Scalar(0, 0, 0), 2, 8, 0, 0.04);

    float angleChange = FindChangeInAngle(abs(suggestedDirection - (prevFrame.cols / 2)), abs(prevFrame.rows - (dims1.getStartRow() + (dims1.getWindowDims().height / 2))));
    cv::putText(prevFrame, "angle: " + std::to_string(angleChange), cv::Point(30, 30), 
        cv::FONT_HERSHEY_PLAIN, 2.5, cv::Scalar(0, 0, 250), 2, CV_MSA);

    ShowImage(prevFrame);
}

// TODO: find gradient of optical flow vector field and label dense large magnitude vector areas as obstacles
void ImageProcessor::ObstacleDetection() const {


}

// TODO: make size of window dependent on width of car and distance from obstacles
int ImageProcessor::FindOptimalDirection(const cv::Mat &opticalFlow, const WindowData &dims) const {

    // separate x and y vector magnitudes and put into pair
    cv::Mat flowParts[2];
    cv::split(opticalFlow, flowParts);
    //std::cout << cv::format(flowParts[0], cv::Formatter::FMT_PYTHON) << '\n';
    std::pair<cv::Mat, cv::Mat> inputPair { std::move(flowParts[0]), std::move(flowParts[1]) };

    // crop, transpoisition, and absolute value fmaps
    class_utils::fmap<decltype(&lFunc)> fmap1 { lFunc };
    class_utils::fmap<decltype(&lTrans)> fmap2 { lTrans };

    // horizontal balance strategy: cut matrix in half -> pick either left or right side -> repeat until 
    // less than window threshold
    int rightColCounter = opticalFlow.cols / 2;
    int leftColCounter = 0;
    int currentWindowSize = rightColCounter;
    while(currentWindowSize != 1) {

        std::cout << "current window size and column counters: " << currentWindowSize << " " << rightColCounter << " " << leftColCounter << '\n';

        // find magnitudes from each side
        float leftMagnitude = SubmatrixCreation(inputPair, fmap1, fmap2, dims.getStartRow(), dims.getStartRow() + dims.getWindowDims().height,
            leftColCounter, rightColCounter);
        float rightMagnitude = SubmatrixCreation(inputPair, fmap1, fmap2, dims.getStartRow(), dims.getStartRow() + dims.getWindowDims().height,
            rightColCounter, rightColCounter + currentWindowSize);

        std::cout << "left and right vector magnitude summations: " << leftMagnitude << " " << rightMagnitude << '\n';

        // go to side with smaller magnitude
        if(leftMagnitude > rightMagnitude) {
            
            leftColCounter = rightColCounter;
            rightColCounter += (currentWindowSize / 2);
            currentWindowSize /= 2;
        }
        else {

            rightColCounter -= (currentWindowSize / 2);
            currentWindowSize /= 2;
        }
    }

    std::cout << "resulting optimal direction: " << rightColCounter << '\n';
    return rightColCounter;
}

// second approach to finding optimal direction
int ImageProcessor::FindOptimalDirection_(const cv::Mat &opticalFlow, const WindowData &dims) const {

    // separate x and y vector magnitudes and put into pair
    cv::Mat flowParts[2];
    cv::split(opticalFlow, flowParts);
    //std::cout << cv::format(flowParts[0], cv::Formatter::FMT_PYTHON) << '\n';
    std::pair<cv::Mat, cv::Mat> inputPair { std::move(flowParts[0]), std::move(flowParts[1]) };

    // crop, transpoisition, and absolute value fmaps
    class_utils::fmap<decltype(&lFunc)> fmap1 { lFunc };
    class_utils::fmap<decltype(&lTrans)> fmap2 { lTrans };

    // other method for finding optimal angle trajectory for car
    int index = 0;
    int optimalDir = -1;
    float minDir = -1;
    while(index != opticalFlow.cols) {

        float outputMag = SubmatrixCreation(inputPair, fmap1, fmap2, dims.getStartRow(), dims.getStartRow() + dims.getWindowDims().height,
            index, index + dims.getWindowDims().width);

        if(minDir == -1 || outputMag < minDir) {

            minDir = outputMag;
            optimalDir = index + (dims.getWindowDims().width / 2);
        }

        index += dims.getWindowDims().width;
    }

    std::cout << "resulting optimal direction (second approach): " << optimalDir << '\n';
    return optimalDir;
}

int ImageProcessor::FindOptimalDirection__(const cv::Mat &opticalFlow, const WindowData &dims) const {

    cv::Mat flowParts[2];
    cv::split(opticalFlow, flowParts);
    std::pair<cv::Mat, cv::Mat> inputPair { std::move(flowParts[0]), std::move(flowParts[1]) };

    Worker<decltype(&lFunc), decltype(&lTrans), cv::Mat> worker { lFunc, lTrans };
    worker.SetInputPair(inputPair);

    int outputDirection = worker.GetOptimalDirection(opticalFlow.cols, dims);
    return outputDirection;
}

// finds change in angle based on new optimal direction
float ImageProcessor::FindChangeInAngle(const size_t horizontalDelta, const size_t verticalDelta) const {

    return atan((float)((float)(horizontalDelta) / verticalDelta)) * (float)(180 / PI);
}