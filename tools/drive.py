import os
import argparse
import logging
import sys
from picamera import PiCamera
import time
from PIL import Image

if __name__ == "__main__":

    # get path to computer vision executable
    parser = argparse.ArgumentParser()
    parser.add_argument("computer_vision", help="path to computer vision executable")
    #parser.add_argument("video", help="path to example input video")
    parser.add_argument("data_location", help="path to where data will be stored")
    args = parser.parse_args()

    # process arguments
    print("input paths given: ", args.computer_vision, " ", args.data_location)
    pathToExe = args.computer_vision
    #pathToVid = args.video
    pathToData = args.data_location

    # capture first and second frame and store at given location
    camera = PiCamera()
    camera.capture(pathToData + "/test_image_frame_one.jpg")
    time.sleep(3) # wait a number of seconds before taking second frame
    camera.capture(pathToData + "/test_image_frame_two.jpg")

    # run executable with input frames and extract output delta angle
    pathToFirstFrame = pathToData + "/test_image_frame_one.jpg"
    pathToSecondFrame = pathToData + "/test_image_frame_two.jpg"
    output_angle = os.system(pathToExe + " " + pathToFirstFrame + " " + pathToSecondFrame)
