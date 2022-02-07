#pragma once

#include <dlib/clustering.h>
#include <dlib/matrix.h>
#include <plot.h>

#include <unordered_map>
#include <experimental/filesystem>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using Coords = std::vector<double>;
using PointCoords = std::pair<Coords, Coords>;
using Clusters = std::unordered_map<size_t, PointCoords>;

class ClusterIdentification {

    public:

        ClusterIdentification(bool);
        ~ClusterIdentification();

        // prep and plotting functions
        void Setup(std::vector<cv::Vec2d>);
        void Setup(const std::string);
        void PlotClusters(const Clusters&, const std::string&, const std::string&);

        // different clustering algorithms available in dlib
        template<typename T>
        void HierchicalClustering(const T&, size_t, const std::string&);

    private:

        plotcpp::Plot *plt;
        
};