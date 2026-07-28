import numpy as np
from scipy.stats import skew, kurtosis, pearsonr
from scipy.fftpack import fft

def get_time_domain_features(signal):
    """Extracts statistical features from the time-domain signal."""
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'rms': np.sqrt(np.mean(np.square(signal))),
        'peak': np.max(np.abs(signal)),
        'skewness': skew(signal),
        'kurtosis': kurtosis(signal),
    }
    return features

def get_har_features(acc_x, acc_y, acc_z):
    """
    Extracts features specific to Human Activity Recognition.
    Expects three arrays representing the X, Y, and Z axes.
    """
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    sma = (np.sum(np.abs(acc_x)) + np.sum(np.abs(acc_y)) + np.sum(np.abs(acc_z))) / len(acc_x)
    corr_xy, _ = pearsonr(acc_x, acc_y)
    corr_xz, _ = pearsonr(acc_x, acc_z)
    corr_yz, _ = pearsonr(acc_y, acc_z)
    iqr = np.percentile(magnitude, 75) - np.percentile(magnitude, 25)
    
    return {
        'mag_mean': np.mean(magnitude),
        'mag_std': np.std(magnitude),
        'mag_iqr': iqr,
        'sma': sma,
        'corr_xy': corr_xy,
        'corr_xz': corr_xz,
        'corr_yz': corr_yz
    }

def get_frequency_domain_features(signal, fs):
    """Extracts features from the frequency-domain using FFT."""
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_values = np.abs(fft(signal))
    
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_values = fft_values[pos_mask]
    
    psd = fft_values**2 / n
    
    if np.sum(psd) == 0:
        return {
            'spectral_centroid': 0,
            'mean_frequency': 0,
            'peak_frequency': 0
        }

    features = {
        'spectral_centroid': np.sum(freqs * psd) / np.sum(psd),
        'mean_frequency': np.mean(freqs * psd) / np.mean(psd),
        'peak_frequency': freqs[np.argmax(psd)]
    }
    return features

def get_gravity_features(gravity_x, gravity_y, gravity_z):
    """Extracts mean features from the gravity signals."""
    return {
        'gravity_mean_x': np.mean(gravity_x),
        'gravity_mean_y': np.mean(gravity_y),
        'gravity_mean_z': np.mean(gravity_z)
    }

# --- New Feature Functions from Experiment ---

def get_gravity_variability_features(gravity_x, gravity_y, gravity_z):
    """Captures the stability of the gravity vector."""
    return {
        'gravity_std_x': np.std(gravity_x),
        'gravity_std_y': np.std(gravity_y),
        'gravity_std_z': np.std(gravity_z)
    }

def get_jerk_features(jerk_x, jerk_y, jerk_z):
    """Extracts basic features from jerk signals."""
    features = {}
    features.update({f"jerk_x_{k}": v for k, v in get_time_domain_features(jerk_x).items()})
    features.update({f"jerk_y_{k}": v for k, v in get_time_domain_features(jerk_y).items()})
    features.update({f"jerk_z_{k}": v for k, v in get_time_domain_features(jerk_z).items()})
    return features

def get_angle_features(acc_mean, gravity_mean):
    """Computes angle between acceleration mean and gravity mean."""
    cos_angle = np.dot(acc_mean, gravity_mean) / (np.linalg.norm(acc_mean) * np.linalg.norm(gravity_mean))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return {'angle_acc_gravity': angle}