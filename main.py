import argparse
import cv2
import numpy as np
import os
import sys
from scipy.signal import butter
from helpers import break_config_file, rgb_to_yiq, yiq_to_rgb, ideal_bandpass


def process(input):
    video = cv2.VideoCapture(input)
    frames = []
    if not video.isOpened():
        print(f"Error opening video file {input}")
        sys.exit(1)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    return np.array(frames), video.get(cv2.CAP_PROP_FPS)


def gen_lap_pyr(image, level):
    laplacian_pyramid = []
    for _ in range(level):
        downsampled_image = cv2.pyrDown(image)
        upsampled_image = cv2.pyrUp(
            downsampled_image, dstsize=(image.shape[1], image.shape[0])
        )
        laplacian_pyramid.append(image - upsampled_image)
        image = downsampled_image
    return laplacian_pyramid


def lap_pyr(images, level):
    laplacian_pyramids = []
    for image in images:
        laplacian_pyramid = gen_lap_pyr(image=rgb_to_yiq(image), level=level)
        laplacian_pyramids.append(laplacian_pyramid)
    return np.asarray(laplacian_pyramids, dtype="object")


def gen_gaus_pyr(image, level):
    image_shape = [image.shape[:2]]
    downsampled_image = image.copy()
    for _ in range(level):
        downsampled_image = cv2.pyrDown(downsampled_image)
        image_shape.append(downsampled_image.shape[:2])
    gaussian_pyramid = downsampled_image
    for _ in range(level):
        gaussian_pyramid = cv2.pyrUp(
            gaussian_pyramid,
        )
    return gaussian_pyramid


def gaus_pyr(images, level):
    gaussian_pyramids = np.zeros_like(images, dtype=np.float32)
    for i in range(len(images)):
        gaussian_pyramids[i] = gen_gaus_pyr(image=rgb_to_yiq(images[i]), level=level)
    return gaussian_pyramids


def filter_laplacian(pyramids, level, fps, freq_range, alpha, lambda_c, exag):
    b_low, a_low = butter(1, freq_range[0], btype="low", output="ba", fs=fps)
    b_high, a_high = butter(1, freq_range[1], btype="low", output="ba", fs=fps)
    filtered_pyramids = np.empty_like(pyramids)
    low = high = pyramids[0]
    filtered_pyramids[0] = pyramids[0]
    delta = lambda_c / 8 / (1 + alpha)
    for i in range(1, len(pyramids)):
        low = (
            -a_low[1] * low + b_low[0] * pyramids[i] + b_low[1] * pyramids[i - 1]
        ) / a_low[0]
        high = (
            -a_high[1] * high + b_high[0] * pyramids[i] + b_high[1] * pyramids[i - 1]
        ) / a_high[0]
        filtered_frame = high - low
        height, width, _ = filtered_frame[0].shape
        lambda_ = np.sqrt(height**2 + width**2) / 3
        curr_alpha = lambda_ / delta / 8 - 1
        curr_alpha = exag * alpha if curr_alpha > alpha else curr_alpha
        for lvl in range(level):
            filtered_frame[lvl][:, :, 0] *= curr_alpha
        filtered_pyramids[i] = filtered_frame
    return filtered_pyramids


def filter_gaussian(pyramids, freq_range, alpha, fps, exag):
    filtered_pyramids = ideal_bandpass(
        images=pyramids, fps=fps, freq_range=freq_range
    ).astype(np.float32)
    filtered_pyramids[:, :, :, 0] *= alpha
    return filtered_pyramids


def reconstruct(image, pyramid, laplacian):
    reconstructed_image = rgb_to_yiq(image)
    if laplacian == 1:
        for level in range(1, pyramid.shape[0] - 1):
            tmp = pyramid[level]
            for _ in range(level):
                tmp = cv2.pyrUp(tmp, dstsize=(tmp.shape[1] * 2, tmp.shape[0] * 2))
            reconstructed_image[:, :, 0] += tmp[:, :, 0].astype(np.float64)
    else:
        reconstructed_image = reconstructed_image + pyramid
    reconstructed_image = yiq_to_rgb(reconstructed_image)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    return reconstructed_image.astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSCI1290 - Final Project")
    parser.add_argument("-vi", "-video_input", help="Input video file")
    parser.add_argument("-o", "-output", help="Output video file")
    parser.add_argument("-c", "-config", help="Config file")
    args = parser.parse_args()
    if not os.path.isfile(args.vi):
        print(f"Error: {args.vi} does not exist")
        sys.exit(1)
    if not os.path.isfile(args.c):
        print(f"Error: {args.c} does not exist")
        sys.exit(1)
    if not args.o:
        print(f"Error: No output file specified")
        sys.exit(1)
    if not args.o.endswith(".mp4"):
        print(f"Error: Output file must be a .mp4 file")
        sys.exit(1)
    if not args.c:
        print(f"Error: No config file specified")
        sys.exit(1)
    config_file = args.c
    config = break_config_file(config_file)
    frames, fps = process(args.vi)
    pyramids = (
        lap_pyr(frames, int(config["levels"]))
        if config["laplacian"]
        else gaus_pyr(frames, int(config["levels"]))
    )
    filtered_pyramids = (
        filter_laplacian(
            pyramids,
            int(config["levels"]),
            fps,
            np.array([config["low"], config["high"]]),
            config["alpha"],
            config["lambda_c"],
            config["exag"],
        )
        if config["laplacian"]
        else filter_gaussian(
            pyramids,
            np.array([config["low"], config["high"]]),
            config["alpha"],
            fps,
            exag=config["exag"],
        )
    )
    final = []
    for i in range(len(frames)):
        final.append(reconstruct(frames[i], filtered_pyramids[i], config["laplacian"]))

    out = cv2.VideoWriter(
        args.o,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (final[0].shape[1], final[0].shape[0]),
    )
    for i in range(len(final)):
        out.write(final[i])
    out.release()
