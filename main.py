import argparse
import cv2
import numpy as np
import os

def process(input, single):
    if single:
        video = cv2.VideoCapture(input)
        if not video.isOpened():
            print("Error opening video file")
            return
    else:
        for video in os.listdir(input):
            if not video.endswith('.mp4') or video.endswith('.avi') or video.endswith('.mov'):
                print("Skipping file: " + video)
            else:
                video = cv2.VideoCapture(input + video)
                if not video.isOpened():
                    print("Error opening video file")
                    return
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSCI1290 - Final Project")
    parser.add_argument('-vi', '-video_input', help="Input video file")
    parser.add_argument('-o', '-output', help="Output video file")
    parser.add_argument('-fi', '-folder_input', help="Input folder")
    args = parser.parse_args()
    if args.vi:
        process(args.vi, True)
        pass
    elif args.fi:
        pass
    else:
        raise argparse.ArgumentTypeError('Must specify either an input video or input folder')