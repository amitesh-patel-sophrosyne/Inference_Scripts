import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from scipy.signal import resample
from ai_preprocessor.src import ECGPipeline

# ----------------- FEATURE EXTRACTION -----------------

def calculate_heart_rate_from_rpeaks(r_peaks, signal_length, fs):
    """
    Calculate heart rate using number of R peaks

    Parameters
    ----------
    r_peaks : array-like
        Sample indices of detected R peaks
    signal_length : int
        Total length of ECG signal (in samples)
    fs : int or float
        Sampling frequency (Hz)

    Returns
    -------
    hr : float
        Heart rate in beats per minute (BPM)
    """

    if r_peaks is None or len(r_peaks) == 0:
        return -1

    duration_sec = signal_length / fs
    num_beats = len(r_peaks)

    hr = (num_beats / duration_sec) * 60
    return int(hr)


def calculate_ecg_features(signal: np.ndarray, unit: str = "mV", fs: int = 360) -> dict:
    """
    Calculate ECG features from a given signal.
    
    Args:
        signal: ECG signal array
        fs: Sampling frequency (default: 360 Hz)
        
    Returns:
        Dictionary containing ECG features
    """
    pipeline = ECGPipeline(target_fs=fs)
    results = pipeline.process(signal, original_fs=fs, unit=unit)
    r_peaks = results.get("r_peaks", [])
    p_peaks = results.get("p_peaks", [])
    pr_intervals = results.get("PR_intervals", [])

    # -------- PR INTERVALS STATISTICS --------
    if len(pr_intervals):
        mean_pr = np.mean(pr_intervals)
        pr_std = np.std(pr_intervals)
        pr_min = np.min(pr_intervals)
        pr_max = np.max(pr_intervals)
    else:
        mean_pr = pr_std = pr_min = pr_max = 0
    
    # -------- HEART RATE --------
    heart_rate = calculate_heart_rate_from_rpeaks(results.get("r_peaks"), len(signal), fs)

    # -------- QRS DURATION --------
    qrs_duration = results.get("mean_qrs", 0)

    # -------- P TO QRS ratio --------
    if len(r_peaks):
        p_to_qrs = len(p_peaks)/len(r_peaks)
    else:
        p_to_qrs = 0

    # -------- ATRIAL RATE --------
    duration_sec = len(signal) / fs
    atrial_rate = (len(p_peaks)/duration_sec)*60

    # -------- RR INTERVALS STATISTICS --------
    rr_intervals = results.get("rr_intervals", [])

    if len(rr_intervals):
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
    else:
        rr_mean = 0
        rr_std = 0


    hrv = results.get("hrv_metrics", {})
    # -------- RMSSD --------
    rmssd = hrv.get('rmssd', 0)
    
    # -------- PRR50 --------
    prr50 = hrv.get('pnn50', 0)
    
    # -------- CVRR --------
    cvrr = hrv.get('cvrr', 0)
    
    return {
        'DETECTED_HR': heart_rate,
        'PR_MEAN': mean_pr,
        'PR_STD': pr_std,
        'QRS_DUR': qrs_duration,
        'P_TO_QRS': p_to_qrs,
        'PR_MIN': pr_min,
        'PR_MAX': pr_max,
        'ATRIAL_RATE': atrial_rate,
        'RR_MEAN': rr_mean,
        'RR_STD': rr_std,
        'RMSSD': rmssd,
        'PRR50': prr50,
        'CVRR': cvrr
    }

# ---------------------- INFERENCE ----------------------------------------

def infer_ecg(model_path: str, scaler_path: str, signal: np.ndarray, unit: str = "mV", original_fs: int = 360) -> tuple[int, float, str]:
    """
    Infer ECG classification from a given signal.
    
    Args:
        model_path: Path to the trained model file
        scaler_path: Path to the scaler file
        signal: ECG signal array
        unit: Unit of the signal (default: "mV")
        original_fs: Sampling frequency (default: 360 Hz)
    
    Returns:
        prediction: int (0, 1, or 2),
        probability: float (0.0 to 1.0),
        disorder_type: str (SINUS BRADYCARDIA, SINUS TACHYCARDIA, or OTHERS)
    """

    # resample the signal to 360Hz
    target_fs = 360
    num_samples = int(len(signal) * target_fs / original_fs)
    signal = resample(signal, num_samples)
    print('RESAMPLING TO 360Hz DONE')


    # LOAD THE MODEL AND SCALER .PKL FILES
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # CALCULATE ECG FEATURES
    FEATURE_COLUMNS = [
    'DETECTED_HR',
    'PR_MEAN',
    'PR_STD',
    'QRS_DUR',
    'P_TO_QRS',
    'PR_MIN',
    'PR_MAX',
    'ATRIAL_RATE',
    'RR_MEAN',
    'RR_STD',
    'RMSSD',
    'PRR50',
    'CVRR'
    ]

    features = calculate_ecg_features(signal, unit, target_fs)
    print('FEATURE EXTRACTION COMPLETED')

    x_test = np.array([[features[col] for col in FEATURE_COLUMNS]])


    # Transform test set using loaded scaler
    x_test_scaled = scaler.transform(x_test)

    # Predict on test set
    pred = int(model.predict(x_test_scaled)[0])
    probs = model.predict_proba(x_test_scaled)[0]
    prob = float(probs[pred])
    print('TESTING COMPLETED')
    
    # CLASS_NAMES = ['SB (0)', 'ST (1)', 'Others (2)']

    if pred == 0:
        disorder_type = 'SINUS BRADYCARDIA'
    elif pred == 1:
        disorder_type = 'SINUS TACHYCARDIA'
    else:
        disorder_type = 'OTHERS'

    return pred, prob, disorder_type

if __name__ == "__main__":

    # ---------------- TEST CONFIG ----------------
    MODEL_PATH = "/Users/amiteshpatel/Desktop/Sophro/IMU_Models/Inference_Scripts/models/arrhythmia/sb_st_model/Random Forest_model.pkl"      # Change to your model path
    SCALER_PATH = "/Users/amiteshpatel/Desktop/Sophro/IMU_Models/Inference_Scripts/models/arrhythmia/sb_st_model/Random Forest_scaler.pkl"    # Change to your scaler path

    fs = 360                      # Sampling frequency (Hz)
    duration = 10                 # seconds
    signal_length = fs * duration # 3600 samples

    # Generate a random signal
    np.random.seed(42)
    signal = np.random.randn(signal_length)

    print(f"Signal length: {len(signal)} samples")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Duration: {duration} seconds")

    try:
        pred, prob, disorder = infer_ecg(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            signal=signal,
            unit="mV",
            original_fs=fs
        )

        print("\n========== RESULT ==========")
        print(f"Prediction ID : {pred}")
        print(f"Probability   : {prob:.4f}")
        print(f"Disorder      : {disorder}")

    except Exception as e:
        print(f"\nInference failed: {e}")