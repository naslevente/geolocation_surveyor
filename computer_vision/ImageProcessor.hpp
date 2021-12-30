#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

class ImageProcessor {

    public:

        ImageProcessor(cv::Mat inputMat);
        ~ImageProcessor();

        // helper functions
        bool GradientSearch(cv::Mat ) const;

        // main functions
        void ShowImage(cv::Mat input) const;
        void EdgeDetection();
        void CornerDetection() const;
        void ImageGradientCalculation(cv::Mat input) const;
        void FindCorners(cv::Mat grad_x, cv::Mat grad_y, cv::Size kernelSize) const;

    private:

        cv::Mat inputMat;
        cv::Mat grayImage;

};


    