"""
Standalone step-counting inference script.

Input : raw wrist accelerometer signal (x, y, z in g), 100 Hz, indexed by timestamp.
Output: total step count + per-window (5s) walk/non-walk label and step count.

Usage:
    total_steps, step_df = infer_steps("signal.csv", "model_bundle.joblib")

model_bundle.joblib must be a dict: {"model": <fitted classifier>, "feature_cols": [...]}
CSV must have columns: timestamp, x, y, z

Deployment notes (left for you to adapt):
    - Swap the CSV reader for your live buffer/stream source.
    - make_windows() currently expects a full recording; for streaming, keep
      calling it on rolling WINDOW_SEC chunks.
"""

import numpy as np
import pandas as pd
import joblib
import scipy.stats as stats
import scipy.signal as signal
import statsmodels.tsa.stattools as stattools

# ── Config ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 100                      # Hz
WINDOW_SEC = 5                         # seconds per inference window
WINDOW_LEN = SAMPLE_RATE * WINDOW_SEC  # samples per window
MIN_WINDOW_SEC = 2                     # min window length required to extract features

PROMINENCE = 0.9
DISTANCE = 0.3 * SAMPLE_RATE


# ── Signal preprocessing ─────────────────────────────────────────────────

def butterfilt(x, cutoff, fs, order=4, axis=0):
    """ Lowpass Butterworth filter """
    nyq = 0.5 * fs
    sos = signal.butter(order, cutoff / nyq, btype='low', analog=False, output='sos')
    return signal.sosfiltfilt(sos, x, axis=axis)


def preprocess_signal(xyz):
    """ Vector magnitude -> remove gravity -> lowpass filter """
    v = np.linalg.norm(xyz, axis=1)
    v = v - 1
    v = butterfilt(v, 5, fs=SAMPLE_RATE)
    return v


# ── Feature extraction (for the walk / non-walk classifier) ─────────────

def extract_features(xyz, sample_rate=SAMPLE_RATE):
    """ HAR time-series features used by the walk classifier """
    if np.isnan(xyz).any() or len(xyz) <= MIN_WINDOW_SEC * sample_rate:
        return {}

    v = preprocess_signal(xyz)

    feats = {}
    feats.update(_moments_features(v))
    feats.update(_quantile_features(v))
    feats.update(_autocorr_features(v, sample_rate))
    feats.update(_spectral_features(v, sample_rate))
    feats.update(_fft_features(v, sample_rate))
    feats.update(_peaks_features(v, sample_rate))
    return feats


def _moments_features(v):
    avg = np.mean(v)
    std = np.std(v)
    if std > .01:
        skew = np.nan_to_num(stats.skew(v))
        kurt = np.nan_to_num(stats.kurtosis(v))
    else:
        skew = kurt = 0
    return {'avg': avg, 'std': std, 'skew': skew, 'kurt': kurt}


def _quantile_features(v):
    feats = {}
    feats['min'], feats['q25'], feats['med'], feats['q75'], feats['max'] = np.quantile(v, (0, .25, .5, .75, 1))
    return feats


def _autocorr_features(v, sample_rate):
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.nan_to_num(stattools.acf(v, nlags=2 * sample_rate))

    peaks, _ = signal.find_peaks(u, prominence=.1)
    if len(peaks) > 0:
        acf_1st_max_loc = peaks[0]
        acf_1st_max = u[acf_1st_max_loc]
        acf_1st_max_loc /= sample_rate
    else:
        acf_1st_max = acf_1st_max_loc = 0.0

    valleys, _ = signal.find_peaks(-u, prominence=.1)
    if len(valleys) > 0:
        acf_1st_min_loc = valleys[0]
        acf_1st_min = u[acf_1st_min_loc]
        acf_1st_min_loc /= sample_rate
    else:
        acf_1st_min = acf_1st_min_loc = 0.0

    acf_zeros = np.sum(np.diff(np.signbit(u)))

    return {
        'acf_1st_max': acf_1st_max,
        'acf_1st_max_loc': acf_1st_max_loc,
        'acf_1st_min': acf_1st_min,
        'acf_1st_min_loc': acf_1st_min_loc,
        'acf_zeros': acf_zeros,
    }


def _spectral_features(v, sample_rate):
    feats = {}
    freqs, powers = signal.periodogram(v, fs=sample_rate, detrend='constant', scaling='density')
    powers /= (len(v) / sample_rate)

    feats['pentropy'] = stats.entropy(powers[powers > 0])
    feats['power'] = np.sum(powers)

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]

    TOPN = 3
    feats.update({f"f{i + 1}": 0 for i in range(TOPN)})
    feats.update({f"p{i + 1}": 0 for i in range(TOPN)})
    for i, j in enumerate(peak_ranks[:TOPN]):
        feats[f"f{i + 1}"] = peak_freqs[j]
        feats[f"p{i + 1}"] = peak_powers[j]

    return feats


def _fft_features(v, sample_rate, nfreqs=5):
    _, powers = signal.welch(
        v, fs=sample_rate, nperseg=sample_rate, noverlap=sample_rate // 2,
        detrend='constant', scaling='density', average='median'
    )
    return {f"fft{i}": powers[i] for i in range(nfreqs + 1)}


def _peaks_features(v, sample_rate):
    feats = {}
    u = butterfilt(v, 5, fs=sample_rate)
    peaks, peak_props = signal.find_peaks(u, distance=0.3 * sample_rate, prominence=0.1)
    feats['npeaks'] = len(peaks) / (len(v) / sample_rate)
    if len(peak_props['prominences']) > 0:
        feats['peaks_avg_promin'] = np.mean(peak_props['prominences'])
        feats['peaks_min_promin'] = np.min(peak_props['prominences'])
        feats['peaks_max_promin'] = np.max(peak_props['prominences'])
    else:
        feats['peaks_avg_promin'] = feats['peaks_min_promin'] = feats['peaks_max_promin'] = 0
    return feats


# ── Step counting ─────────────────────────────────────────────────────────

def count_steps(xyz):
    """ Lowpass filter + normalize, then count peaks in the window """
    filtered = preprocess_signal(xyz)
    norm = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    peaks, _ = signal.find_peaks(norm, distance=DISTANCE, prominence=PROMINENCE)
    return len(peaks)


# ── Windowing + main inference ────────────────────────────────────────────

def make_windows(data):
    """ Split the signal into non-overlapping WINDOW_SEC windows and extract features """
    rows, raw_windows = [], []
    for t, w in data.resample(f"{WINDOW_SEC}s"):
        if len(w) < WINDOW_LEN:
            continue
        xyz = w[['x', 'y', 'z']].to_numpy()
        rows.append({'time': t, **extract_features(xyz)})
        raw_windows.append(xyz)

    frame = pd.DataFrame(rows)
    raw_windows = np.stack(raw_windows)
    return raw_windows, frame


def infer_steps(csv_path, model_path):
    """
    Run full inference on a CSV of raw accelerometer data.

    csv_path   : path to CSV with columns [timestamp, x, y, z]
    model_path : path to joblib bundle {"model": clf, "feature_cols": [...]}

    Returns:
        total_steps : int
        step_df     : per-window DataFrame with columns [time, is_walk, steps]
    """
    artifact = joblib.load(model_path)
    clf = artifact["model"]
    feature_cols = artifact["feature_cols"]

    data = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')

    raw_windows, frame = make_windows(data)
    X = frame[feature_cols].to_numpy()
    is_walk_pred = clf.predict(X)

    total_steps = 0
    step_timeline = []

    for i, (xyz, is_walk) in enumerate(zip(raw_windows, is_walk_pred)):
        steps = count_steps(xyz) if is_walk else 0
        total_steps += steps
        step_timeline.append({'time': frame.iloc[i]['time'], 'is_walk': int(is_walk), 'steps': steps})

    step_df = pd.DataFrame(step_timeline)
    return total_steps, step_df


if __name__ == "__main__":
    import sys
    csv_path = "/Users/amiteshpatel/Desktop/Sophro/IMU_Models/stepcount/data/OxWalk_Dec2022/Wrist_100Hz/P02_wrist100.csv"
    model_path = "/Users/amiteshpatel/Desktop/Sophro/IMU_Models/Inference_Scripts/models/imu/step-count-10-59-43/model_bundle.joblib"
    total_steps, step_df = infer_steps(csv_path, model_path)
    print(f"Total steps: {total_steps}")
    # print(step_df)