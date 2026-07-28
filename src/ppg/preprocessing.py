import numpy as np
from scipy.signal import cheby2, filtfilt, butter
import pywt
import matplotlib.pyplot as plt

# ----------------------------------- General Preprocessing -----------------------------------

def moving_average_filter1(signal, fs, window_sec=0.6):

    window = int(fs * window_sec)
    
    kernel = np.ones(window) / window
    
    pad_size = window // 2
    
    # Reflect padding
    padded_signal = np.pad(signal, pad_size, mode='reflect')
    
    # Convolution
    filtered = np.convolve(padded_signal, kernel, mode='same')
    
    # Remove padding
    filtered = filtered[pad_size:-pad_size]
    
    return filtered

def remove_baseline_wander1(raw_signal, baseline):
    """
    Remove baseline wander from PPG signal

    Parameters
    ----------
    raw_signal : numpy array
        Original PPG signal (x[n])
    baseline : numpy array
        Estimated baseline signal (y[n])

    Returns
    -------
    corrected_signal : numpy array
        Baseline removed PPG signal (z[n])
    """

    corrected_signal = raw_signal - baseline

    return corrected_signal


def chebyshev_type2_lowpass(signal, fs, cutoff=10, order=7, rs=40):
    """
    Apply Chebyshev Type-II Low Pass Filter

    Parameters
    ----------
    signal : numpy array
        Input PPG signal (baseline removed signal)
    fs : float
        Sampling frequency (Hz)
    cutoff : float
        Cutoff frequency (Hz) (default = 10 Hz)
    order : int
        Filter order (default = 7)
    rs : float
        Stopband attenuation in dB (default = 40)

    Returns
    -------
    filtered_signal : numpy array
        Low-pass filtered signal
    """

    # Nyquist frequency
    nyquist = fs / 2

    # Normalized cutoff frequency
    normal_cutoff = cutoff / nyquist

    # Design Chebyshev Type-II filter
    b, a = cheby2(order, rs, normal_cutoff, btype='low', analog=False)

    # Apply zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


# --------------------- Motion artifact removal Preprocessing ------------------------------

def butter_bandpass(signal, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band", analog=False)
    bandpass_filter_signal = filtfilt(b, a, signal)
    return bandpass_filter_signal


    
def get_all_swt_levels(signal, fs, wavelet='db4', level=3):
    signal = np.array(signal)

    n = len(signal)
    next_pow2 = int(2**np.ceil(np.log2(n)))

    if n != next_pow2:
        signal = np.pad(signal, (0, next_pow2 - n), mode='constant')

    current_signal = signal.copy()
    coeffs = pywt.swt(signal, wavelet, level=6)
    
    A6, D6 = coeffs[0][0], coeffs[0][1]
    A5, D5 = coeffs[1][0], coeffs[1][1]
    A4, D4 = coeffs[2][0], coeffs[2][1]
    A3, D3 = coeffs[3][0], coeffs[3][1]
    A2, D2 = coeffs[4][0], coeffs[4][1]
    A1, D1 = coeffs[5][0], coeffs[5][1]

    A6 = A6[:len(signal)]

    A6_filt = butter_bandpass(A6, lowcut=0.5, highcut=3, fs=fs)

    D1 = np.zeros_like(D1)
    D2 = np.zeros_like(D2)
    D3 = np.zeros_like(D3)
    D4 = np.zeros_like(D4)
    D5 = np.zeros_like(D5)
    D6 = np.zeros_like(D6)

    # Reconstruct properly using ISWT
    coeffs_modified = [
        (A6_filt, D6),
        (A5, D5),
        (A4, D4),
        (A3, D3),
        (A2, D2),
        (A1, D1)
    ]

    reconstructed_signal = pywt.iswt(coeffs_modified, wavelet)
    

    return A1, D1, A2, D2, A3, D3, A4, D4, A5, D5, A6, D6, A6_filt, reconstructed_signal
