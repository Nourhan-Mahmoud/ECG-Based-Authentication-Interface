import matplotlib.pyplot as plt
import wfdb
from scipy import signal
import scipy
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.signal import butter, filtfilt
import pywt
import pandas as pd
fs = 1000
# region reading Data


def read_data(path):
    fs = 1000
    patient = wfdb.rdrecord(path, channels=[1])
    signal = patient.p_signal[:, 0]
    time = len(signal) / fs
    return signal, time, patient
# endregion

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

# region helper functions


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


def process_signal(denoised_signal, y_lfiltered):
    # Reduce all values close to zero to be zero
    y_lfiltered[y_lfiltered < 0.1] = 0

    # Get Q and S
    qx = []
    qy = []
    sx = []
    sy = []
    qrs_state = False

    for i in range(len(y_lfiltered)):
        if not qrs_state:
            if y_lfiltered[i] != 0:
                qx.append(i)
                qy.append(denoised_signal[i])
                qrs_state = True
        else:
            if y_lfiltered[i] == 0:
                sx.append(i)
                sy.append(denoised_signal[i])
                qrs_state = False

    # Remove invalid QRS (Incomplete QRS)
    if qrs_state:
        idx = qx.pop()
        for i in range(idx, len(y_lfiltered)):
            y_lfiltered[i] = 0
        qy.pop()

    # Fix s and q (Crawl towards correct value)
    for i in range(len(qx)):
        idx = qx[i]
        while True:
            if denoised_signal[idx - 1] < denoised_signal[idx]:
                qx[i] = idx - 1
                qy[i] = denoised_signal[idx - 1]
            else:
                break
            idx -= 1
        idx = qx[i]
        while True:
            if denoised_signal[idx + 1] < denoised_signal[idx]:
                qx[i] = idx + 1
                qy[i] = denoised_signal[idx + 1]
            else:
                break
            idx += 1

    for i in range(len(sx)):
        idx = sx[i]
        while True:
            if denoised_signal[idx - 1] < denoised_signal[idx]:
                sx[i] = idx - 1
                sy[i] = denoised_signal[idx - 1]
            else:
                break
            idx -= 1
        idx = sx[i]
        while True:
            if denoised_signal[idx + 1] < denoised_signal[idx]:
                sx[i] = idx + 1
                sy[i] = denoised_signal[idx + 1]
            else:
                break
            idx += 1

    qx = np.array(qx)
    qy = np.array(qy)
    sx = np.array(sx)
    sy = np.array(sy)

    return qx, qy, sx, sy


def process_qrs(denoised_signal, y_lfiltered, window_smoothed_signal, qx, qy, sx, sy):
    # Thresholding (Get R, any value not the peak between Q and S is set to 0)
    for i in range(len(qx)):
        y_lfiltered[qx[i]:sx[i]][y_lfiltered[qx[i]:sx[i]]
                                 != max(y_lfiltered[qx[i]:sx[i]])] = 0

    # Remove any peaks that are not Rs
    y_lfiltered[y_lfiltered < max(y_lfiltered) * 0.6] = 0

    # Retrieve R
    Rx = []
    Ry = []
    for i in range(len(y_lfiltered)):
        if y_lfiltered[i] != 0:
            Rx.append(i)
            Ry.append(denoised_signal[i])

    # Fix R (Crawl to correct value)
    for i in range(len(Rx)):
        idx = Rx[i]
        while True:
            if denoised_signal[idx - 1] > denoised_signal[idx]:
                Rx[i] = idx - 1
                Ry[i] = denoised_signal[idx - 1]
            else:
                break
            idx -= 1
        idx = Rx[i]
        while True:
            if denoised_signal[idx + 1] > denoised_signal[idx]:
                Rx[i] = idx + 1
                Ry[i] = denoised_signal[idx + 1]
            else:
                break
            idx += 1

    Rx = np.array(Rx)
    Ry = np.array(Ry)

    # Get QRS onset and offset
    qrs_off_x = []
    qrs_on_x = []
    qrs_off_y = []
    qrs_on_y = []
    for x in Rx:
        X = np.array([x, window_smoothed_signal[x]])
        Y = np.array([x - 300, window_smoothed_signal[x - 300]])
        qrs_on_x.append(get_onset_offset(X, Y, window_smoothed_signal))
        qrs_on_y.append(denoised_signal[qrs_on_x[-1]])

        Y = np.array([x + 200, window_smoothed_signal[x + 200]])
        qrs_off_x.append(get_onset_offset(X, Y, window_smoothed_signal))
        qrs_off_y.append(denoised_signal[qrs_off_x[-1]])

    qrs_on_x = np.array(qrs_on_x)
    qrs_off_x = np.array(qrs_off_x)
    qrs_on_y = np.array(qrs_on_y)
    qrs_off_y = np.array(qrs_off_y)
    return qx, qy, sx, sy, Rx, Ry, qrs_on_x, qrs_on_y, qrs_off_x, qrs_off_y


def extract_p_wave(fs, qrs_on_x, denoised_signal):
    # Get P wave
    window_size = int(fs * 0.2)  # 200 ms window
    Px = []
    Py = []
    for loc in qrs_on_x:
        start_idx = int(loc - window_size)

        px = 0
        py = -5
        for i in range(start_idx, loc):
            if denoised_signal[i] > py:
                py = denoised_signal[i]
                px = i
        Py.append(py)
        Px.append(px)
    Px = np.array(Px)
    Py = np.array(Py)
    return Px, Py


def calculate_p_onset_offset(Px, denoised_signal):
    window_size = int(fs * 0.2)  # 200 ms window
    p_on_x = []
    p_off_x = []
    p_on_y = []
    p_off_y = []

    for x in Px:
        X = np.array([x, denoised_signal[x]])
        Y = np.array([x - window_size, denoised_signal[x - window_size]])
        p_on_x.append(get_onset_offset(X, Y, denoised_signal))
        p_on_y.append(denoised_signal[p_on_x[-1]])

        Y = np.array([x + 50, denoised_signal[x + 50]])
        p_off_x.append(get_onset_offset(X, Y, denoised_signal))
        p_off_y.append(denoised_signal[p_off_x[-1]])

    p_on_x = np.array(p_on_x)
    p_off_x = np.array(p_off_x)
    p_on_y = np.array(p_on_y)
    p_off_y = np.array(p_off_y)

    return p_on_x, p_off_x, p_on_y, p_off_y


def calculate_t_wave(qrs_off_x, denoised_signal):
    window_size = int(fs * 0.5)  # 400 ms window
    Tx = []
    Ty = []

    for loc in qrs_off_x:
        tx = 0
        ty = 5
        loc += 10
        start_idx = int(loc + window_size)
        for i in range(loc, start_idx):
            if(i+50 < len(denoised_signal)):
                w1 = denoised_signal[i - 50] - denoised_signal[i]

                w2 = denoised_signal[i] - denoised_signal[i + 50]
                w = w1 * w2
                if w < ty:
                    ty = w
                    tx = i

        Tx.append(tx)
        Ty.append(denoised_signal[tx])
    Tx = np.array(Tx)
    Ty = np.array(Ty)
    return Tx, Ty


def calculate_t_onset_offset(Tx, denoised_signal):
    t_off_x = []
    t_on_x = []
    t_off_y = []
    t_on_y = []

    for x in Tx:
        X = np.array([x, denoised_signal[x]])
        Y = np.array([x - 200, denoised_signal[x - 200]])
        t_on_x.append(get_onset_offset(X, Y, denoised_signal))
        t_on_y.append(denoised_signal[t_on_x[-1]])
        if(x+150 < len(denoised_signal)):
            Y = np.array([x + 200, denoised_signal[x + 200]])
        else:
            Y = np.array([len(denoised_signal)-1, denoised_signal[-1]])
        t_off_x.append(get_onset_offset(X, Y, denoised_signal))
        t_off_y.append(denoised_signal[t_off_x[-1]])

    t_on_x = np.array(t_on_x)
    t_off_x = np.array(t_off_x)
    t_on_y = np.array(t_on_y)
    t_off_y = np.array(t_off_y)

    return t_on_x, t_off_x, t_on_y, t_off_y


def plot_signals_with_t(denoised_signal, y_lfiltered, qrs_on_x, qrs_on_y, qrs_off_x, qrs_off_y,
                        qx, qy, sx, sy, Rx, Ry, Px, Py, p_on_x, p_on_y, p_off_x, p_off_y,
                        Tx, Ty, t_on_x, t_on_y, t_off_x, t_off_y, time, fs):
    ts = np.arange(0, time, 1.0 / fs)  # time vector

    fig = plt.figure(figsize=[15, 10.4])
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    fig.suptitle("Signals")
    axs[0].plot(ts, denoised_signal, alpha=0.6, lw=1, label="Raw signal")
    axs[1].plot(ts, y_lfiltered, alpha=0.6, lw=1, label="SciPy lfilter")

    axs[0].scatter(qrs_on_x/fs, qrs_on_y, color="yellow", s=7)
    axs[0].scatter(qrs_off_x/fs, qrs_off_y, color="lime", s=7)
    axs[0].scatter(qx/fs, qy, color="red", s=7)
    axs[0].scatter(sx/fs, sy, color="green", s=7)
    axs[0].scatter(Rx/fs, Ry, color="blue", s=7)

    axs[0].scatter(Px/fs, Py, color="#17becf", s=7)
    axs[0].scatter(p_on_x/fs, p_on_y, color="#e377c2", s=7)
    axs[0].scatter(p_off_x/fs, p_off_y, color="black", s=7)

    axs[0].scatter(Tx/fs, Ty, color="#bcbd22", s=7)
    axs[0].scatter(t_on_x/fs, t_on_y, color="orange", s=7)
    axs[0].scatter(t_off_x/fs, t_off_y, color="brown", s=7)

    plt.show()


def calculate_non_fiducial(Rx, denoised_signal):
    RR_previous = Rx[1] - Rx[0]
    RR_next = Rx[2] - Rx[1]

    nonFiducial = []

    after_Rpeak = int(2/3 * ((RR_previous + RR_next) / 2))

    for i in range(int(Rx[1]), int(Rx[1]) + after_Rpeak):
        nonFiducial.append(denoised_signal[i])

    Before_Rpeak = int(1/3 * ((RR_previous + RR_next) / 2))

    for i in range(Before_Rpeak, int(Rx[1])):
        nonFiducial.append(denoised_signal[i])
   # plt.plot(nonFiducial)
    return nonFiducial


def extract_non_fiducial_feature(nonFiducial):
    # Define the mother wavelet
    wavelet = pywt.Wavelet('db4')

    # Define the number of levels for decomposition
    decomp_levels = 5

    # Define the list of ECG signals
    ecg_segments = np.array(nonFiducial)  # Example data

    # Decompose the signal
    decomp = pywt.wavedec(ecg_segments, wavelet, level=decomp_levels)

    CA5, CD5, CD4, CD3, CD2, CD1 = decomp

    # Only the coefficients of ECG band (1-40) use them as a feature
    non_fiducial_feature = CA5[:41]

    # plt.plot(non_fiducial_feature)
    # plt.show()

    return non_fiducial_feature


def create_dataframe(qx, qy, sx, sy, Rx, Ry, qrs_on_x, qrs_on_y, qrs_off_x, qrs_off_y, Px, Py, p_on_x, p_off_x, p_on_y, p_off_y, Tx, Ty, t_on_x, t_off_x, t_on_y, t_off_y):
    data = {
        'qx': qx,
        'qy': qy,
        'sx': sx,
        'sy': sy,
        'Rx': Rx,
        'Ry': Ry,
        'qrs_on_x': qrs_on_x,
        'qrs_on_y': qrs_on_y,
        'qrs_off_x': qrs_off_x,
        'qrs_off_y': qrs_off_y,
        'Px': Px,
        'Py': Py,
        'p_on_x': p_on_x,
        'p_off_x': p_off_x,
        'p_on_y': p_on_y,
        'p_off_y': p_off_y,
        'Tx': Tx,
        'Ty': Ty,
        't_on_x': t_on_x,
        't_off_x': t_off_x,
        't_on_y': t_on_y,
        't_off_y': t_off_y
    }

    df = pd.DataFrame(data)
    return df

# endregion

# region Feature Detection


def Feature_Detection(signal):
    fs = 1000
    denoised_signal, y_lfiltered, window_smoothed_signal = processing(signal)
    time = len(signal)/fs
    qx, qy, sx, sy = process_signal(denoised_signal, y_lfiltered)
    qx, qy, sx, sy, Rx, Ry, qrs_on_x, qrs_on_y, qrs_off_x, qrs_off_y = process_qrs(denoised_signal, y_lfiltered, window_smoothed_signal,
                                                                                   qx, qy, sx, sy)
    Px, Py = extract_p_wave(fs, qrs_on_x, denoised_signal)
    p_on_x, p_off_x, p_on_y, p_off_y = calculate_p_onset_offset(
        Px, denoised_signal)
    Tx, Ty = calculate_t_wave(qrs_off_x, denoised_signal)
    t_on_x, t_off_x, t_on_y, t_off_y = calculate_t_onset_offset(
        Tx, denoised_signal)

    Fiducial_Points = create_dataframe(qx, qy, sx, sy, Rx, Ry, qrs_on_x, qrs_on_y, qrs_off_x, qrs_off_y,
                                       Px, Py, p_on_x, p_off_x, p_on_y, p_off_y, Tx, Ty, t_on_x, t_off_x, t_on_y, t_off_y)
    nonFiducial = calculate_non_fiducial(Rx, denoised_signal)
    non_fiducial_feature = extract_non_fiducial_feature(nonFiducial)
    #plot_signals_with_t(denoised_signal, y_lfiltered, qrs_on_x, qrs_on_y, qrs_off_x, qrs_off_y,qx, qy, sx, sy, Rx, Ry, Px, Py, p_on_x, p_on_y, p_off_x, p_off_y,Tx, Ty, t_on_x, t_on_y, t_off_x, t_off_y, time, fs)
    return Fiducial_Points, non_fiducial_feature
# endregion
