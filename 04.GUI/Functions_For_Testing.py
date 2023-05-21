import wfdb
from scipy import signal
import scipy
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd
import pywt
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from scipy.signal import butter, filtfilt

fs = 1000

# region Preproccessing


def butter_bandpass_filter(input_signal, low_cutoff, high_cutoff, sampling_rate, order):
    nyq = 0.5 * sampling_rate
    low = low_cutoff / nyq
    high = high_cutoff / nyq
    numerator, denominator = butter(
        order, [low, high], btype='band', output='ba', analog=False, fs=None)
    filtered = filtfilt(numerator, denominator, input_signal)
    return filtered


def smoothMAconv(depth, temp, scale):  # Moving average by numpy convolution
    dz = np.diff(depth)
    N = int(scale/dz[0])
    smoothed = np.convolve(temp, np.ones((N,))/N, mode='same')
    return smoothed


def get_onset_offset(X, Y, signal):
    a_norm = math.sqrt((Y[0]-X[0])**2 + (Y[1]-X[1])**2)

    a = np.array([[X[0], X[1]], [Y[0], Y[1]]])

    c_x = X[0]
    prev_sigma_max = -1
    max_x = -1
    while True:
        if X[0] > Y[0]:
            if c_x <= Y[0]:
                return max_x
            c_x -= 1
        else:
            if c_x >= Y[0]:
                return max_x
            c_x += 1

        c = np.array([[X[0], X[1]], [c_x, signal[int(c_x)]]])

        ac_cross = np.cross(a, c)
        m_cross = (a[0][0]-a[1][0]) * (c[0][1]-c[1][1]) - \
            (a[0][1]-a[1][1]) * (c[0][0]-c[1][0])

        ac_norm = np.linalg.norm(ac_cross)
        sigma = ac_norm / a_norm
        if X[0] > Y[0]:
            sigma = m_cross
        else:
            sigma = -m_cross

        if sigma > prev_sigma_max:
            prev_sigma_max = sigma
            max_x = int(c_x)
# endregion


def processing(signal):
    # STEPS 1 to 4
    # 1. Bandpass (low pass / high pass)
    y_lfiltered = butter_bandpass_filter(
        signal, low_cutoff=1.0, high_cutoff=40.0, sampling_rate=1000, order=2)
    denoised_signal = y_lfiltered

    # 2. Differentiation
    y_lfiltered = np.gradient(y_lfiltered)

    # 3. Squaring
    y_lfiltered = y_lfiltered ** 2

    # 4. Window smoothing
    n = 40
    y_lfiltered = np.convolve(y_lfiltered, np.ones((n,))/n, mode='same')

    # Resize for next processing
    y_lfiltered = y_lfiltered * 1000
    window_smoothed_signal = y_lfiltered.copy()
    return denoised_signal, y_lfiltered, window_smoothed_signal
