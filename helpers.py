import numpy as np


def break_config_file(config_file):
    config = {}
    with open(config_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            key, value = line.strip().split(", ")
            config[key] = float(value)
    return config


def rgb_to_yiq(rgb_image):
    transformation_matrix = np.array(
        [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]]
    )

    yiq_image = np.dot(rgb_image, transformation_matrix.T)
    return yiq_image


def yiq_to_rgb(yiq_image):
    transformation_matrix = np.array(
        [[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.106, 1.703]]
    )

    rgb_image = np.dot(yiq_image, transformation_matrix.T)
    return rgb_image


def ideal_bandpass(images, fps, freq_range, axis=0):
    fft = np.fft.fft(images, axis=axis)
    frequencies = np.fft.fftfreq(images.shape[axis], d=1.0 / fps)

    low = (np.abs(frequencies - freq_range[0])).argmin()
    high = (np.abs(frequencies - freq_range[1])).argmin()

    fft[:low, :] = 0
    fft[high:, :] = 0

    return np.fft.ifft(fft, axis=axis).real
