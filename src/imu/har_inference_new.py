"""
har_inference.py
=================

Deployment-time inference for the HAR (Human Activity Recognition) model
trained in train.py. Loads the saved model/scaler/feature-columns and turns
a raw sensor window (or a CSV stream of windows) into an activity prediction.

WHY THIS FILE EXISTS
---------------------
The training pipeline computes a "gravity" signal as `total_acc - body_acc`,
which only works because the UCI HAR dataset ships both signals pre-split.
In the real world your device gives you ONE accelerometer stream, and it
either:

  (a) still contains gravity   -> e.g. Android TYPE_ACCELEROMETER, most raw
                                   IMU/accelerometer chips, iOS `userAcceleration`
                                   is NOT this case (see below).
  (b) has gravity already removed -> e.g. Android TYPE_LINEAR_ACCELERATION,
                                      iOS CMDeviceMotion.userAcceleration.

This script exposes a `gravity_mode` argument so you tell it which one you
have, instead of guessing:

  gravity_mode="raw"
      The accelerometer still contains gravity. We reproduce the UCI HAR
      dataset's own method to split it: a median filter + 3rd-order low-pass
      Butterworth filter (20 Hz) to denoise, then ANOTHER 3rd-order low-pass
      Butterworth filter (0.3 Hz corner frequency) to isolate the gravity
      (near-DC) component. body_acc = raw_acc - gravity.

  gravity_mode="linear"
      Gravity has already been physically removed by the device/OS. There is
      no way to recover it from the acc signal alone (the information is
      gone), so you have two sub-options:
        - pass `gravity_xyz=(gx, gy, gz)` if the device also exposes a
          dedicated gravity sensor (Android TYPE_GRAVITY) sampled over the
          same window -> used as-is, no filtering.
        - pass nothing -> gravity-derived features (gravity_mean_*,
          gravity_std_*, angle_acc_gravity) fall back to zero. This is a
          documented, safe default: predictions still work, but accuracy on
          activities that gravity features help disambiguate (e.g.
          SITTING vs STANDING) may be somewhat degraded versus training-time
          performance. Recommend testing this fallback against a held-out
          set before relying on it in production.

Everything else (jerk signals, time/frequency-domain stats, correlations)
is computed with the *exact* same formulas as train.py, on purpose -- the
model only knows the feature distribution it was trained on, so inference
must reproduce that distribution bug-for-bug (e.g. jerk here is a plain
np.diff, not divided by dt, because that's what train.py did).

USAGE AS A LIBRARY
-------------------
    from har_inference import HARPredictor

    predictor = HARPredictor(model_dir="models/exp0", fs=50)

    result = predictor.predict_window(
        acc_x, acc_y, acc_z,      # each: 128 raw accelerometer samples
        gyro_x, gyro_y, gyro_z,   # each: 128 raw gyroscope samples
        gravity_mode="raw",       # or "linear"
        return_proba=True,
    )
    print(result)
    # {'activity_id': 4, 'activity': 'STANDING', 'probabilities': {...}}

USAGE AS A CLI (continuous CSV sensor stream -> sliding-window predictions)
-----------------------------------------------------------------------
    python har_inference.py \\
        --model-dir models/exp0 \\
        --input-csv sensor_stream.csv \\
        --gravity-mode raw \\
        --output-csv predictions.csv

    sensor_stream.csv must have columns: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt, medfilt

from feature_extraction import (
    get_time_domain_features,
    get_har_features,
    get_frequency_domain_features,
    get_gravity_features,
    get_gravity_variability_features,
    get_jerk_features,
    get_angle_features,
)

# 0-indexed to match train.py's `y["Activity"] - 1`
ACTIVITY_LABELS = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}

GRAVITY_CUTOFF_HZ = 0.3   # UCI HAR: gravity assumed to be the near-DC component
NOISE_CUTOFF_HZ = 20.0    # UCI HAR: noise-removal low-pass corner frequency
VALID_GRAVITY_MODES = ("raw", "linear")


class HARPredictor:
    """Loads a trained HAR model bundle and predicts an activity per window."""

    def __init__(self, model_dir, fs=50):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "best_har_model.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.feature_columns = joblib.load(model_dir / "feature_columns.pkl")
        self.fs = fs

    # ---------------------------------------------------------------
    # Gravity handling
    # ---------------------------------------------------------------
    @staticmethod
    def _butter_lowpass(data, cutoff, fs, order=3):
        nyq = 0.5 * fs
        wn = cutoff / nyq
        if not (0 < wn < 1):
            raise ValueError(
                f"Cutoff {cutoff}Hz is invalid for sampling rate {fs}Hz "
                f"(normalized cutoff {wn:.3f} must be in (0, 1))."
            )
        b, a = butter(order, wn, btype="low")
        padlen = 3 * max(len(a), len(b))
        if len(data) <= padlen:
            raise ValueError(
                f"Window of {len(data)} samples is too short to filter "
                f"(need > {padlen} samples for a stable {order}rd-order "
                f"Butterworth filter). Use a longer window or a lower filter order."
            )
        return filtfilt(b, a, data)

    def _split_gravity_from_raw(self, raw_axis, apply_noise_filter=True):
        """Reproduce the UCI HAR method: denoise, then isolate the gravity
        (near-DC) component with a 0.3Hz low-pass filter. body = raw - gravity.
        """
        sig = np.asarray(raw_axis, dtype=float)
        if apply_noise_filter:
            k = 3
            if len(sig) >= k:
                sig = medfilt(sig, kernel_size=k)
            sig = self._butter_lowpass(sig, NOISE_CUTOFF_HZ, self.fs, order=3)
        gravity = self._butter_lowpass(sig, GRAVITY_CUTOFF_HZ, self.fs, order=3)
        body = sig - gravity
        return body, gravity

    @staticmethod
    def _broadcast(value, n):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr))
        if len(arr) != n:
            raise ValueError(f"Gravity array length {len(arr)} does not match window length {n}.")
        return arr

    def _resolve_body_and_gravity(self, acc_x, acc_y, acc_z, gravity_mode,
                                   gravity_xyz=None, apply_noise_filter=True):
        if gravity_mode == "raw":
            bx, gx = self._split_gravity_from_raw(acc_x, apply_noise_filter)
            by, gy = self._split_gravity_from_raw(acc_y, apply_noise_filter)
            bz, gz = self._split_gravity_from_raw(acc_z, apply_noise_filter)
            return (bx, by, bz), (gx, gy, gz)

        elif gravity_mode == "linear":
            n = len(acc_x)
            body = (
                np.asarray(acc_x, dtype=float),
                np.asarray(acc_y, dtype=float),
                np.asarray(acc_z, dtype=float),
            )
            if gravity_xyz is not None:
                gx, gy, gz = gravity_xyz
                gravity = (self._broadcast(gx, n), self._broadcast(gy, n), self._broadcast(gz, n))
            else:
                warnings.warn(
                    "gravity_mode='linear' with no gravity_xyz supplied: "
                    "gravity-derived features will be zeroed. This is a safe "
                    "fallback but may reduce accuracy vs. training-time performance.",
                    stacklevel=2,
                )
                gravity = (np.zeros(n), np.zeros(n), np.zeros(n))
            return body, gravity

        else:
            raise ValueError(f"gravity_mode must be one of {VALID_GRAVITY_MODES}, got '{gravity_mode}'.")

    # ---------------------------------------------------------------
    # Feature extraction (mirrors train.py's extract_features_from_signals)
    # ---------------------------------------------------------------
    def _extract_features(self, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
                           gravity_mode, gravity_xyz=None, apply_noise_filter=True):
        (body_acc_x, body_acc_y, body_acc_z), (gravity_x, gravity_y, gravity_z) = \
            self._resolve_body_and_gravity(acc_x, acc_y, acc_z, gravity_mode,
                                            gravity_xyz, apply_noise_filter)

        # Gyroscope has no gravity component - used directly, same as training.
        body_gyro_x = np.asarray(gyro_x, dtype=float)
        body_gyro_y = np.asarray(gyro_y, dtype=float)
        body_gyro_z = np.asarray(gyro_z, dtype=float)

        # Jerk: identical formula to train.py (plain np.diff, NOT divided by
        # dt) so the feature distribution matches what the model learned.
        body_acc_jerk_x = np.diff(body_acc_x, prepend=body_acc_x[0])
        body_acc_jerk_y = np.diff(body_acc_y, prepend=body_acc_y[0])
        body_acc_jerk_z = np.diff(body_acc_z, prepend=body_acc_z[0])
        body_gyro_jerk_x = np.diff(body_gyro_x, prepend=body_gyro_x[0])
        body_gyro_jerk_y = np.diff(body_gyro_y, prepend=body_gyro_y[0])
        body_gyro_jerk_z = np.diff(body_gyro_z, prepend=body_gyro_z[0])

        features = {}
        features.update({f"acc_{k}": v for k, v in get_time_domain_features(body_acc_x).items()})
        features.update({f"acc_{k}": v for k, v in get_har_features(body_acc_x, body_acc_y, body_acc_z).items()})
        features.update({f"acc_{k}": v for k, v in get_frequency_domain_features(body_acc_x, self.fs).items()})

        features.update({f"gyro_{k}": v for k, v in get_time_domain_features(body_gyro_x).items()})
        features.update({f"gyro_{k}": v for k, v in get_har_features(body_gyro_x, body_gyro_y, body_gyro_z).items()})
        features.update({f"gyro_{k}": v for k, v in get_frequency_domain_features(body_gyro_x, self.fs).items()})

        features.update(get_gravity_features(gravity_x, gravity_y, gravity_z))
        features.update(get_gravity_variability_features(gravity_x, gravity_y, gravity_z))

        features.update({f"body_acc_{k}": v for k, v in get_jerk_features(body_acc_jerk_x, body_acc_jerk_y, body_acc_jerk_z).items()})
        features.update({f"body_gyro_{k}": v for k, v in get_jerk_features(body_gyro_jerk_x, body_gyro_jerk_y, body_gyro_jerk_z).items()})

        acc_mean_vec = [np.mean(body_acc_x), np.mean(body_acc_y), np.mean(body_acc_z)]
        gravity_mean_vec = [np.mean(gravity_x), np.mean(gravity_y), np.mean(gravity_z)]

        # Guard divide-by-zero in angle calc (norm=0 happens when gravity
        # fell back to all-zeros in 'linear' mode with no gravity supplied).
        if np.linalg.norm(gravity_mean_vec) == 0 or np.linalg.norm(acc_mean_vec) == 0:
            features["angle_acc_gravity"] = 0.0
        else:
            features.update(get_angle_features(acc_mean_vec, gravity_mean_vec))

        return features

    # ---------------------------------------------------------------
    # Public prediction API
    # ---------------------------------------------------------------
    def predict_window(self, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
                        gravity_mode="raw", gravity_xyz=None,
                        apply_noise_filter=True, return_proba=False):
        """Predict the activity for a single window of raw sensor samples.

        acc_x/y/z, gyro_x/y/z: array-like, same length (e.g. 128 samples).
        gravity_mode: "raw" (accelerometer contains gravity) or
                      "linear" (gravity already removed by the device).
        gravity_xyz: optional (gx, gy, gz) arrays/scalars, only used when
                     gravity_mode="linear" and a dedicated gravity sensor
                     reading is available.
        """
        feats = self._extract_features(
            acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
            gravity_mode=gravity_mode, gravity_xyz=gravity_xyz,
            apply_noise_filter=apply_noise_filter,
        )
        X = pd.DataFrame([feats])
        # Reindex to the exact columns/order seen at training time; any
        # feature the model doesn't know about is dropped, anything missing
        # (shouldn't happen, but just in case) is filled with 0.
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)

        pred = int(self.model.predict(X_scaled)[0])
        result = {"activity_id": pred, "activity": ACTIVITY_LABELS.get(pred, str(pred))}

        if return_proba:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X_scaled)[0]
                result["probabilities"] = {
                    ACTIVITY_LABELS.get(i, str(i)): float(p) for i, p in enumerate(proba)
                }
            else:
                warnings.warn(f"{type(self.model).__name__} has no predict_proba; skipping probabilities.")

        return result

    def predict_stream(self, df, window_size=128, step_size=64,
                        gravity_mode="raw", gravity_df=None,
                        apply_noise_filter=True, return_proba=False):
        """Slide a window over a continuous sensor stream and predict each window.

        df: DataFrame with columns acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z.
        gravity_df: optional DataFrame (same index range as df) with columns
                    gravity_x, gravity_y, gravity_z, used only when
                    gravity_mode="linear".
        """
        required_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Input stream is missing required columns: {missing}")

        results = []
        n = len(df)
        for start in range(0, max(n - window_size + 1, 0), step_size):
            end = start + window_size
            window = df.iloc[start:end]

            gravity_xyz = None
            if gravity_df is not None:
                gwin = gravity_df.iloc[start:end]
                gravity_xyz = (
                    gwin["gravity_x"].values,
                    gwin["gravity_y"].values,
                    gwin["gravity_z"].values,
                )

            res = self.predict_window(
                window["acc_x"].values, window["acc_y"].values, window["acc_z"].values,
                window["gyro_x"].values, window["gyro_y"].values, window["gyro_z"].values,
                gravity_mode=gravity_mode, gravity_xyz=gravity_xyz,
                apply_noise_filter=apply_noise_filter, return_proba=return_proba,
            )
            res["window_start"] = start
            res["window_end"] = end
            results.append(res)

        return pd.json_normalize(results)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run HAR inference on a continuous CSV sensor stream using a sliding window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True,
                         help="Directory with best_har_model.pkl, scaler.pkl, feature_columns.pkl")
    parser.add_argument("--input-csv", required=True,
                         help="CSV with columns: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z")
    parser.add_argument("--fs", type=int, default=50, help="Sampling frequency in Hz")
    parser.add_argument("--window-size", type=int, default=128, help="Samples per window (128 = 2.56s @ 50Hz)")
    parser.add_argument("--step-size", type=int, default=64, help="Step between windows (64 = 50% overlap)")
    parser.add_argument(
        "--gravity-mode", choices=VALID_GRAVITY_MODES, default="raw",
        help="'raw': accelerometer still contains gravity, separated via the UCI HAR "
             "Butterworth method. 'linear': gravity already removed by the device.",
    )
    parser.add_argument(
        "--gravity-csv", default=None,
        help="Optional CSV with gravity_x, gravity_y, gravity_z columns (e.g. from a "
             "dedicated gravity sensor). Only used with --gravity-mode=linear. If omitted "
             "in linear mode, gravity-based features fall back to zero.",
    )
    parser.add_argument("--no-noise-filter", action="store_true",
                         help="Skip the median+low-pass noise filter before gravity separation (raw mode only).")
    parser.add_argument("--output-csv", default=None, help="Write predictions here. Also prints to stdout.")
    parser.add_argument("--proba", action="store_true", help="Include per-class probabilities in the output.")
    return parser


def main():
    args = _build_arg_parser().parse_args()

    predictor = HARPredictor(args.model_dir, fs=args.fs)

    df = pd.read_csv(args.input_csv)
    gravity_df = pd.read_csv(args.gravity_csv) if args.gravity_csv else None

    out_df = predictor.predict_stream(
        df,
        window_size=args.window_size,
        step_size=args.step_size,
        gravity_mode=args.gravity_mode,
        gravity_df=gravity_df,
        apply_noise_filter=not args.no_noise_filter,
        return_proba=args.proba,
    )

    if args.output_csv:
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(out_df)} predictions to {args.output_csv}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()