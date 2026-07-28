import numpy as np
import math
import pywt

def trim_signals_to_length(reference_signal, *signals):
    """
    Trim all input signals to the length of reference_signal.

    Returns
    -------
    tuple
        Trimmed signals in the same order.
    """

    target_length = len(reference_signal)

    trimmed_signals = tuple(
        signal[:target_length] for signal in signals
    )

    return trimmed_signals

def upper_lower_threshold(signal, peaks, valleys):
    
    peak_amp = []
    valley_amp = []
    
    for i in range(len(peaks)):
        peak_amp.append(signal[peaks[i]])
        valley_amp.append(signal[valleys[i]])
    
    upper_thres = np.median(peak_amp) + 2 * np.std(peak_amp)
    lower_thres = np.median(valley_amp) - 2 * np.std(valley_amp)
    
    return upper_thres, lower_thres

def extract_motion_artifact_signal(signal, lower_threshold, upper_threshold):
    """
    Extract motion artifact signal by zeroing values
    that lie within the threshold range.

    Returns
    -------
    numpy.ndarray
        Motion artifact extracted signal.
    """

    motion_artifact_signal = np.copy(signal)

    for i, sig in enumerate(motion_artifact_signal):

        # Values inside threshold range -> set to 0
        if lower_threshold < sig < upper_threshold:
            motion_artifact_signal[i] = 0

    return motion_artifact_signal


def remove_motion_artifact(
    filtered_signal,
    reconstructed_signal,
    approximation_coeffs,
    detail_coeffs,
    motion_artifact_approximation,
    wavelet='db4'
):
    """
    Reconstruct motion artifact signal using inverse SWT
    and subtract it from reconstructed signal.

    Returns
    -------
    motion_artifact_sig : ndarray
        Reconstructed motion artifact signal.

    motion_artifact_removed : ndarray
        Signal after motion artifact removal.
    """

    # Original signal length
    orig_len = len(filtered_signal)

    # SWT requires power-of-2 length
    padded_len = int(2 ** math.ceil(math.log2(orig_len)))

    def pad_to_length(arr, target_len):
        """Pad array to target length with zeros."""
        if len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)), mode='constant')
        return arr

    # Unpack coefficients
    A1, A2, A3, A4, A5 = approximation_coeffs
    D1, D2, D3, D4, D5, D6 = detail_coeffs

    # Create modified coefficient list for inverse SWT
    coeffs_modified = [
        (pad_to_length(motion_artifact_approximation, padded_len), pad_to_length(D6, padded_len)),
        (pad_to_length(A5, padded_len), pad_to_length(D5, padded_len)),
        (pad_to_length(A4, padded_len), pad_to_length(D4, padded_len)),
        (pad_to_length(A3, padded_len), pad_to_length(D3, padded_len)),
        (pad_to_length(A2, padded_len), pad_to_length(D2, padded_len)),
        (pad_to_length(A1, padded_len), pad_to_length(D1, padded_len)),
    ]

    # Reconstruct motion artifact signal
    motion_artifact_sig = pywt.iswt(coeffs_modified, wavelet=wavelet)

    # Trim back to original length
    motion_artifact_sig = motion_artifact_sig[:orig_len]

    # Remove motion artifact
    motion_artifact_removed = reconstructed_signal - motion_artifact_sig

    return motion_artifact_sig, motion_artifact_removed
    