#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

class ImageProcessor {

    public:

        ImageProcessor(cv::Mat inputMat);
        ImageProcessor();
        ~ImageProcessor();

        // helper functions
        template<typename T, typename ...U>
        bool SubmatrixCreation(std::pair<T, T> inputGrads, U ...args) const;

        // main functions
        void ShowImage(const cv::Mat &input) const;
        void LocateObstacle();
        void CornerDetection() const;
        void ImageGradientCalculation(cv::Mat input, cv::Mat &grad_x, cv::Mat &grad_y) const;
        void ProcessGradients(cv::Mat grad_x, cv::Mat grad_y, cv::Size kernelSize) const;
        void OpticalFlowCalculation(cv::Mat &prevFrame, cv::Mat &currentFrame, cv::Mat &outputFrame) const;

    private:

        cv::Mat inputMat;
        cv::Mat grayImage;

};


    