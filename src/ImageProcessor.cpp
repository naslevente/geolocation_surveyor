#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <cassert>
#include <numeric>

#include "ImageProcessor.hpp"
#include "class_utils.hpp"

ImageProcessor::ImageProcessor(cv::Mat inputMat) {

    this->inputMat = std::move(inputMat);
}

ImageProcessor::~ImageProcessor() = default;

void ImageProcessor::ShowImage(cv::Mat input) const {

    cv::imshow("output image", input);
    cv::waitKey();
}

// 2 fmap inputs: row/col manipulation, transposition
template<typename T, typename ...U>
bool ImageProcessor::SubmatrixCreation(std::pair<T, T> inputGrads, U ...args) const {

    // unpack fmap and two integer values
    auto params = std::make_tuple(args...);

    // apply crop and transposition lambdas on input gradient pair
    auto newPair = std::get<0>(params)(inputGrads, std::get<2>(params), std::get<3>(params));
    newPair = std::get<1>(params)(inputGrads);
    newPair = std::get<0>(params)(inputGrads, std::get<4>(params), std::get<5>(params));

    // check if kernel window has both x and y non-zero gradients (a.k.a it is corner)
    auto accumulation = [](cv::Mat input) -> bool {

        int finalSum = 0;
        for(int i = 0; i < input.rows; ++i) {

            const auto row = input.row(i);
            finalSum += std::accumulate(row.begin<int>(), row.end<int>(), 0);
        }

        return finalSum > 0;
    };

    return (accumulation(std::get<0>(newPair)) && accumulation(std::get<1>(newPair)));
}

void ImageProcessor::EdgeDetection() {

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
    this->grayImage = std::move(grayscale);

    // calculate image gradients
    this->ImageGradientCalculation(finalImage);
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
    //ShowImage(finalNormalized);

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

void ImageProcessor::ImageGradientCalculation(cv::Mat inputMat) const {

    // define constants and other necessary variables
    const int kernelSize = 3;
    const int scale = 1; // default value
    const int delta = 0; // default value
    const int depth = CV_16S;
    cv::Mat grad_x;
    cv::Mat grad_y;

    // calculate gradients with sobel operator
    cv::Sobel(inputMat, grad_x, depth, 1, 0, kernelSize, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(inputMat, grad_y, depth, 0, 1, kernelSize, scale, delta, cv::BORDER_DEFAULT);

    // call FindCorners function on newly calculated gradients
    this->FindCorners(grad_x, grad_y, cv::Size(11, 11));

    //std::cout << "x gradient output: " << '\n' << cv::format(grad_x, cv::Formatter::FMT_PYTHON) << '\n';
    //std::cout << "y gradient output: " << '\n' << cv::format(grad_x, cv::Formatter::FMT_PYTHON) << '\n';

}

void ImageProcessor::FindCorners(cv::Mat grad_x, cv::Mat grad_y, cv::Size kernelSize) const { // kernel size dimensions must be odd

    // transposition and crop lambda functions applied to two fmaps that will be sent to SubmatrixCreation function
    auto lFunc = [](cv::Mat input, int param1, int param2) -> cv::Mat {

        return input.rowRange(param1, param2);
    };
    class_utils::fmap<decltype(lFunc)> fmap1 {std::move(lFunc)};

    auto lTrans = [](cv::Mat input) -> cv::Mat {

        return input.t();
    };
    class_utils::fmap<decltype(lTrans)> fmap2 {std::move(lTrans)};

    // define constants
    const int row_const = ((kernelSize.height - 1) / 2);
    const int col_const = ((kernelSize.width - 1) / 2);

    // define iterating and other necessary variables
    int rowCounter = kernelSize.height / 2;
    int colCounter = kernelSize.width / 2;
    int rowIterations = grad_x.rows / kernelSize.height;
    int colIteration = grad_x.cols / kernelSize.width;
    std::pair<cv::Mat, cv::Mat> inputPair {grad_x, grad_y};

    // iterate through both image gradients
    for(int i = 0; i < rowIterations; ++i) {

        // assign value based on whether or not kernel exceeds rows
        int row_param2 = rowCounter + row_const > grad_x.rows ? grad_x.rows : rowCounter + row_const;

        for(int j = 0; j < colIteration; ++j) {

            // assign value based on whether or not kernel exceeds cols
            int col_param2 = colCounter + col_const > grad_x.cols ? grad_x.cols : colCounter + col_const;

            if(SubmatrixCreation(inputPair, fmap1, fmap2, rowCounter - row_const, row_param2, colCounter - col_const, col_param2)) {

                const auto xVal = (j * kernelSize.width) - col_const;
                const auto yVal = (i * kernelSize.height) - row_const;
                circle(this->inputMat, cv::Point(yVal, xVal), 1,  cv::Scalar(0), 2, 8, 0);
            }

            colCounter += kernelSize.width;
        }

        colCounter = kernelSize.width / 2;
        rowCounter += kernelSize.height;
    }
}
