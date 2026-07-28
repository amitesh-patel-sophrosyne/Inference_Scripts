import os
import joblib
import numpy as np


# Prediction labels
# 0 = ADL
# 1 = FALL

# Level-1 gatekeeper thresholds
CUM_W_THRESHOLD = 32.3357
MU_ACCY_THRESHOLD = 0.162

# converting SisFall's raw ADC counts into physical units using the specifications of the sensors used in the dataset
# ADXL345 → ±16 g, 8192 LSB/g
# ITG3200 → 14.375 LSB/(°/s)
def _extract_features(acc, gyro):
    # Convert to physical units (same as training)
    acc_x = acc[:, 0] * (32.0 / 8192.0)
    acc_y = acc[:, 1] * (32.0 / 8192.0)
    acc_z = acc[:, 2] * (32.0 / 8192.0)

    gyro_x = gyro[:, 0] / 14.375
    gyro_y = gyro[:, 1] / 14.375
    gyro_z = gyro[:, 2] / 14.375

    acc_svm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    peak_idx = np.argmax(acc_svm)
    start = max(0, peak_idx - 30)
    end = min(len(acc_svm), peak_idx + 30)

    acc_x = acc_x[start:end]
    acc_y = acc_y[start:end]
    acc_z = acc_z[start:end]

    gyro_x = gyro_x[start:end]
    gyro_y = gyro_y[start:end]
    gyro_z = gyro_z[start:end]

    acc_svm = acc_svm[start:end]

    features = {}

    channels = {
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "ACCsvm": acc_svm,
    }

    for name, signal in channels.items():
        features[f"{name}_mean"] = np.mean(signal)
        features[f"{name}_max"] = np.max(signal)
        features[f"{name}_min"] = np.min(signal)
        features[f"{name}_std"] = np.std(signal)
        features[f"{name}_range"] = np.max(signal) - np.min(signal)
        features[f"{name}_var"] = signal[-1] - signal[0]

    features["SMA"] = np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z))
    features["ACCsvm_avg_rate"] = np.mean(np.abs(np.diff(acc_svm)))
    features["mu_ACCy"] = features["acc_y_mean"]

    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    features["cum_w"] = np.trapz(gyro_mag, dx=1 / 200)

    return features


def predict(acc, gyro, model_dir):
    """
    Parameters
    ----------
    acc : ndarray (N,3)
        Raw ADXL345 accelerometer samples.

    gyro : ndarray (N,3)
        Raw ITG3200 gyroscope samples.

    model_dir : str
        Directory containing:
            model.joblib
            scaler.joblib
            features.txt

    Returns
    -------
    {
        "prediction": 0 or 1,
        "probability": float
    }
    """

    acc = np.asarray(acc, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must have shape (N,3)")

    if gyro.ndim != 2 or gyro.shape[1] != 3:
        raise ValueError("gyro must have shape (N,3)")

    if len(acc) != len(gyro):
        raise ValueError("acc and gyro must have the same number of samples")

    features = _extract_features(acc, gyro)

    # Level-1 gatekeeper
    if (
        features["cum_w"] <= CUM_W_THRESHOLD
        and abs(features["mu_ACCy"]) <= MU_ACCY_THRESHOLD
    ):
        return {
            "prediction": 0,
            "probability": 0.0,
        }

    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

    with open(os.path.join(model_dir, "features.txt")) as f:
        feature_names = [line.strip() for line in f if line.strip()]

    x = np.array([[features[name] for name in feature_names]], dtype=np.float32)

    x = scaler.transform(x)

    prediction = int(model.predict(x)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(x)[0][1])
    else:
        probability = None

    return {
        "prediction": prediction,
        "probability": probability,
    }


if __name__ == "__main__":
    # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    acc = np.random.randint(-2000, 2000, (600, 3))
    gyro = np.random.randint(-500, 500, (600, 3))

    result = predict(
        acc,
        gyro,
        model_dir="/Users/amiteshpatel/Desktop/Sophro/IMU_Models/Inference_Scripts/models/imu/fall_detection_xgboost_2026-04-17_14-42-24",
    )

    print(result)