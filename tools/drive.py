import os
import argparse
import logging
import sys
from picamera import PiCamera
import time
from PIL import Image

#Motor and proximity sensor imports
import signal
import RPi.GPIO as MotorGPIO
import RPi.GPIO as ProximityGPIO

#Servo imports
from gpiozero import Servo

#**ESC ARMING IS A MUST BEFORE RUNNING THIS SCRIPT**
if __name__ == "__main__":
    #Setting proximity sensor configs
    ProximityGPIO.setmode(ProximityGPIO.BCM)
    ProximityGPIO.setup(25, ProximityGPIO.IN)

    #Setting up motor configs
    MotorGPIO.setwarnings(False)
    MotorGPIO.setmode(MotorGPIO.BCM)
    MotorGPIO.setup(27, MotorGPIO.OUT)
    MotorPWM = MotorGPIO.PWM(27, 50)


    #Setting up servos
    steer_servo = Servo(27)
    break_servo = Servo(17)

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

    idx = 0

    try:

        while True:
            # **Do NOT forget to arm ESC before this**
            # TODO: enter here motor starting/continuing to work
            MotorPWM.start(0)
            MotorPWM.ChangeDutyCycle(8)

            # capture first and second frame and store at given location
            camera = PiCamera()
            camera.capture(pathToData + "/test_image_frame_" + idx + ".jpg")
            pathToFirstFrame = pathToData + "/test_image_frame_" + idx + ".jpg"
            print("captured first frame | " + pathToFirstFrame)
            idx = idx + 1

            time.sleep(3) # wait a number of seconds before taking second frame
            camera.capture(pathToData + "/test_image_frame_" + idx + ".jpg")
            pathToSecondFrame = pathToData + "/test_image_frame_" + idx + ".jpg"
            print("captured second frame | " + pathToSecondFrame)
            idx = idx + 1

            # run executable with input frames and extract output delta angle
            output_angle = os.system(pathToExe + " " + pathToFirstFrame + " " + pathToSecondFrame)

            #Proximity sensor output
            if GPIO.input(25):
                print("NO COLLISION")
            else:
                print("COLLISION")
            time.sleep(0.5)

            # TODO: enter here change in steering based off output
            steer_servo.value = ANGLE_BETWEEN_1_AND_MINUS_1
            #time.sleep(3) or however long it should wait until next capture

            #Breaking servo activated
            break_servo.value = -0.5

            #Stopping motor
            MotorPWM.ChangeDutyCycle(0)

    except KeyboardInterrupt:
        #Cleanup GPIO config
        GPIO.cleanup()
        print("---Geolocation Surveyor test run ended---")
