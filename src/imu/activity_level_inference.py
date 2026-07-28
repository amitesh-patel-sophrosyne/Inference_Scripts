import pickle
import numpy as np


def predict(signal, model_path="logistic_regression.pkl", source_fs=100, target_fs=50):
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2 or signal.shape[1] != 3:
        raise ValueError("signal must have shape (N,3)")

    if source_fs != target_fs:
        if source_fs % target_fs != 0:
            raise ValueError("source_fs must be an integer multiple of target_fs")
        signal = signal[::source_fs // target_fs]

    svm = np.sqrt(np.sum(signal.astype(np.float64) ** 2, axis=1))

    dsvm = np.zeros_like(svm)
    dsvm[1:] = svm[1:] - svm[:-1]

    features = np.array([[
        np.mean(svm),
        np.std(svm),
        np.mean(np.abs(dsvm)),
        np.std(dsvm)
    ]], dtype=np.float32)

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    features = model_data["poly"].transform(features)
    features = model_data["scaler"].transform(features)

    return int(model_data["model"].predict(features)[0])


if __name__ == "__main__":
    signal = np.random.randn(600, 3).astype(np.float32)
    level = predict(signal, model_path="/Users/amiteshpatel/Desktop/Sophro/IMU_Models/Inference_Scripts/models/imu/activity_level/logistic_regression.pkl")
    print(level)