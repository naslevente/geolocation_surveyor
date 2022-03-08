import os
import argparse
import logging
import sys

if __name__ == "__main__":

    # get path to computer vision executable
    parser = argparse.ArgumentParser();
    parser.add_argument("computer_vision", help="path to computer vision executable")
    parser.add_argument("video", help="path to example input video")
    args = parser.parse_args();

    # process arguments
    print("input paths given: ", args.computer_vision, " ", args.video)
    pathToExe = args.computer_vision
    pathToVid = args.video

    # run executable with input video
    os.system(pathToExe + " " + pathToVid)