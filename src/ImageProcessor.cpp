#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <cassert>
#include <numeric>

#include "ImageProcessor.hpp"

ImageProcessor::ImageProcessor(cv::Mat inputMat) {

    this->inputMat = std::move(inputMat);
}

ImageProcessor::~ImageProcessor() = default;

void ImageProcessor::ShowImage(cv::Mat input) const {

    cv::imshow("output image", input);
    cv::waitKey();
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

    // define constants
    int rowIteration = kernelSize.height / 2;
    int colIteration = kernelSize.width / 2;

    // iterate through both image gradients
    for(int i = 0; i < grad_x.rows; ++i) {

        for(int j = 0; j < grad_x.cols; ++j) {

            const int row_const = ((kernelSize.height - 1) / 2);
            const int col_const = ((kernelSize.width - 1) / 2);
            // check each neighboring gradient value to determine if corner
            auto submatrix_row = grad_x.rowRange(rowIteration - row_const, rowIteration + row_const);
            auto submatrix_col = submatrix_row.colRange(rowIteration - col_const, rowIteration + col_const);

            auto finalSum = 0;
            for(int k = 0; k < submatrix_col.rows; ++k) {

                const auto submatrix = submatrix_col.row(k);
                finalSum += std::accumulate(submatrix.begin<int>(), submatrix.end<int>(), 0);
            }

            colIteration += kernelSize.width;
        }

        colIteration = kernelSize.width / 2;
        rowIteration += kernelSize.height;
    }
}
