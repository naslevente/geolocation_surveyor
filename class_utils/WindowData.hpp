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

class WindowData {

    public:

        WindowData(cv::Size inputDims, int startRow) {

            this->startRow = startRow;
            this->windowDims = inputDims;
        }

        ~WindowData() = default;
        
        cv::Size getWindowDims() {

            return windowDims;
        }

        cv::Size& setWindowDims() {

            return windowDims;
        }

        int getStartRow() {

            return startRow;
        }

        int& setStartRow() {

            return startRow;
        }
    
    private:

        cv::Size windowDims;
        int startRow;
};
