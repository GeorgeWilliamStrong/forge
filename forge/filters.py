import numpy as np
import scipy


def butter_filter(data, threshold, fs, btype='lowpass', order=5):
    """
    Apply a zero-phase Butterworth filter to the input data.

    Parameters
    ----------
    data : array_like
        The input data to be filtered.
    threshold : float
        The cutoff frequency of the filter in Hz. It must be between 0 and
        `fs/2`, where `fs` is the sampling frequency.
    fs : float
        The sampling frequency of the data.
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter. Default is 'lowpass'.
    order : int, optional
        The order of the filter. Default is 5.

    Returns
    -------
    filtered_data : ndarray
        The filtered data.

    """
    nyq = 0.5 * fs
    thresh = threshold / nyq

    b, a = scipy.signal.butter(order, thresh, btype=btype)

    return np.array(scipy.signal.filtfilt(b, a, data), dtype=np.float32)


def butter_lowpass(cutoff, fs, order=5):
    """
    Design a lowpass Butterworth filter.

    Parameters
    ----------
    cutoff : float
        The cutoff frequency of the filter in Hz.
    fs : float
        The sampling frequency of the data.
    order : int, optional
        The order of the filter. Default is 5.

    Returns
    -------
    b : ndarray
        The numerator coefficients of the filter.
    a : ndarray
        The denominator coefficients of the filter.

    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)

    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a lowpass Butterworth filter to the input data.

    Parameters
    ----------
    data : array_like
        The input data to be filtered.
    cutoff : float
        The cutoff frequency of the filter in Hz.
    fs : float
        The sampling frequency of the data.
    order : int, optional
        The order of the filter. Default is 5.

    Returns
    -------
    filtered_data : ndarray
        The filtered data.

    """
    b, a = butter_lowpass(cutoff, fs, order=order)

    return np.asarray(scipy.signal.lfilter(b, a, data), dtype=np.float32)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass Butterworth filter to the input data.

    Parameters
    ----------
    data : array_like
        The input data to be filtered.
    lowcut : float
        The low cutoff frequency of the filter in Hz.
    highcut : float
        The high cutoff frequency of the filter in Hz.
    fs : float
        The sampling frequency of the data.
    order : int, optional
        The order of the filter. Default is 5.

    Returns
    -------
    filtered_data : ndarray
        The filtered data.

    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = scipy.signal.butter(order, [low, high], analog=False, btype='band',
                              output='sos')

    return np.array(scipy.signal.sosfilt(sos, data), dtype=np.float32)
