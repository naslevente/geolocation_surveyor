#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "WindowData.hpp"

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
        void ShowImage(const cv::Mat &input) const;
        void ShowImage(const cv::Mat &input, const cv::Mat &input2) const;
        void RemoveNoise(cv::Mat &inputMat) const;
        template<typename T, typename ...U>
        float SubmatrixCreation(std::pair<T, T> &inputPair, U ...args) const;

        // fmap functions
        static cv::Mat lFunc(const cv::Mat &input, int param1, int param2);
        static cv::Mat lTrans(const cv::Mat &dinput);

        // deprecated approach
        void LocateObstacle();
        void CornerDetection() const;
        void ImageGradientCalculation(cv::Mat input, cv::Mat &grad_x, cv::Mat &grad_y) const;
        void ProcessGradients(cv::Mat grad_x, cv::Mat grad_y, cv::Size kernelSize) const;

        // optical flow approach
        void OpticalFlowCalculation(cv::Mat &prevFrame, cv::Mat &currentFrame, cv::Mat &outputFrame) const;
        void ObstacleDetection() const;
        int FindOptimalDirection(cv::Mat &opticalFlow, WindowData dims) const;
        int FindOptimalDirection_(cv::Mat &opticalFlow, WindowData dims) const;
        float FindChangeInAngle(const size_t horizontalDelta, const size_t verticalDelta) const;

    private:

        cv::Mat inputMat;
        cv::Mat grayImage;

};


    