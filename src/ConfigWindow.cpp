#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>

int main(int argc, char *argv[]) {

    // command line parsing
    argparse::ArgumentParser parser("Geolocation Surveyor");
    parser.add_argument("first_frame").help("<first frame>.jpg/png: path to first input frame");

    try {

        parser.parse_args(argc, argv);
    } catch(const std::runtime_error &err) {

        std::cerr << err.what() << '\n';
        exit(-1);
    }

    auto pathToFirstFrame = parser.get<std::string>("first_frame");
    cv::Mat outputFrame = cv::imread(pathToFirstFrame);

    // add subwindow lines and output image to view setup
    cv::line(prevFrame, cv::Point(0, 50), cv::Point(prevFrame.cols, 50), cv::Scalar(0, 0, 0), 2);
    cv::line(prevFrame, cv::Point(0, 200), cv::Point(prevFrame.cols, 200), cv::Scalar(0, 0, 0), 2);
    cv::imshow("output image", outputFrame);
    cv::waitKey();
}