"""
Standalone deployment inference script for the SVT DINO model.

Pipeline: raw single-lead (II) ECG signal
          -> preprocess (denoise + resample, RR/HR/PSD features, DINO image)
          -> ECGDinoModel (+ PEFT/LoRA adapter "SVT")
          -> alpha-scaled softmax + threshold
          -> {SVT} prob & binary prediction

This script has no dependency on the training repo other than:
  - data_utils.ai_preprocessor.src.ECGPreprocessing (ECGPreprocessor, ECGPipeline)
  - models/dino_model_feat.py (bundled alongside this script)

Usage:
    python infer_dino_svt.py --checkpoint /path/to/adapter/SVT \
                              --signal-npy /path/to/signal.npy \
                              --fs 500 --unit mV
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.signal import welch
from scipy.special import softmax
from transformers import AutoImageProcessor
from peft import PeftModel

# from ECGPreprocessing import (
#     ECGPreprocessor,
#     ECGPipeline,
# )
# from models.dino_model_feat import ECGDinoConfig, ECGDinoModel
from ai_preprocessor.src.ECGPreprocessing import (
    ECGPreprocessor,
)
from ai_preprocessor.src.pipeline import ECGPipeline
from dino_model_feat import ECGDinoConfig, ECGDinoModel



# --------------------------------------------------------------------------
# Fixed config for this head (matches training checkpoint)
# --------------------------------------------------------------------------
IMAGE_MODEL_NAME = "facebook/dinov2-small"
NUM_FEATURES = 6
NUM_LABELS = 2  # binary: [negative, SVT]

BEST_ALPHA = 0.6000000000000001
BEST_THRESHOLD = 0.38

TARGET_FS = 1000  # ECGPreprocessor resampling target


# --------------------------------------------------------------------------
# Image generation (verbatim from training pipeline)
# --------------------------------------------------------------------------
def ecg_to_image_large_fix_nozscore_dino(
    signal, width=500, height=100, line_color=255, bg_color=0
):
    """
    Convert a 1D ECG signal to a grayscale PIL image, resized for the DINO
    processor. No pixel-value normalization is applied here -- that is
    handled by AutoImageProcessor.
    """
    signal = np.array(signal)
    signal = signal - np.min(signal)
    signal = signal / np.max(signal)
    signal = height - (signal * height)

    if len(signal) > width:
        signal = signal[:width]
    else:
        signal = np.pad(signal, (0, width - len(signal)), mode="edge")

    img = Image.new("L", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    for i in range(len(signal) - 1):
        draw.line((i, signal[i], i + 1, signal[i + 1]), fill=line_color)

    compressed_img = img.resize((720, 50), Image.Resampling.LANCZOS)

    return compressed_img


# --------------------------------------------------------------------------
# Handcrafted PSD features (verbatim from ECGDatasetType19)
# --------------------------------------------------------------------------
def extract_psd_features_dynamic(ecg_signal, fs):
    n_seg = int(fs * 1.024)
    freqs, psd = welch(ecg_signal, fs=fs, nperseg=n_seg)

    atrial_mask = (freqs >= 3) & (freqs <= 9)

    if atrial_mask.sum() > 0:
        daf = freqs[atrial_mask][np.argmax(psd[atrial_mask])]
    else:
        daf = 0.0

    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]

    if total_power > 0:
        roll_off_idx = np.where(cumulative_power >= 0.85 * total_power)[0][0]
        roll_off = freqs[roll_off_idx]
    else:
        roll_off = 0.0

    return np.array([daf, roll_off], dtype=np.float32)


# --------------------------------------------------------------------------
# Alpha / threshold post-processing (verbatim from notebook)
# --------------------------------------------------------------------------
def apply_alpha_threshold(logits, alpha, threshold):
    """
    logits: (N, 2)
    alpha: scaling factor for positive class logit
    threshold: probability threshold for positive class
    """
    scaled_logits = logits.copy()
    scaled_logits[:, 1] *= alpha

    probs = softmax(scaled_logits, axis=1)[:, 1]
    preds = (probs >= threshold).astype(int)

    return preds, probs


# --------------------------------------------------------------------------
# Preprocessing: raw signal -> (pixel_values, features)
# --------------------------------------------------------------------------
class ECGPreprocessPipeline:
    """
    Wraps everything needed to go from a raw single-lead signal to the
    tensors the DINO model expects. Mirrors ECGDatasetType19.__getitem__
    and generate_dataset()'s R-peak/HR extraction.
    """

    def __init__(self, target_fs=TARGET_FS):
        self.target_fs = target_fs
        self.preprocessor = ECGPreprocessor(target_fs=target_fs)
        self.image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)

    def _dino_transform(self, image):
        inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        return inputs["pixel_values"][0]

    def __call__(self, signal, fs, unit="mV"):
        """
        signal: 1D array-like, raw single-lead (II) ECG signal
        fs: sampling rate of `signal` (Hz)
        unit: signal unit, e.g. "mV"

        Returns:
            pixel_values: torch.FloatTensor (1, 3, H, W)
            features: torch.FloatTensor (1, 6)
        """
        if unit.lower() == "mv":
            unit = "mV"

        # raw_signal = torch.tensor(np.asarray(signal), dtype=torch.float32)
        raw_signal = np.asarray(signal, dtype=np.float32)


        # ---- R-peaks / HR (computed on the raw, native-fs signal) ----
        try:
            pipeline = ECGPipeline(target_fs=fs)
            results = pipeline.process(
                raw_signal, original_fs=fs, unit=unit
            )
            r_peaks = results["r_peaks"]
            hr = results["mean_heart_rate_bpm"]
        except Exception as e:
            print(f"[warning] delineation pipeline failed: {e}")
            r_peaks = []
            hr = -1

        rr_intervals = (
            np.diff(r_peaks) / fs if len(r_peaks) > 1 else np.array([0.0])
        )
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
        rr_diff = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0

        rr_mean = 0.0 if np.isnan(rr_mean) else rr_mean
        rr_std = 0.0 if np.isnan(rr_std) else rr_std
        rmssd = 0.0 if np.isnan(rmssd) else rmssd

        # ---- denoise + resample to target_fs ----
        normalized = self.preprocessor.normalizing_ecg(raw_signal, unit, fs)
        smoothed, *_ = self.preprocessor.denoise_ecg(normalized)
        smoothed_t = torch.tensor(smoothed, dtype=torch.float32)

        if not torch.isfinite(smoothed_t).all():
            raise ValueError("NaNs/Infs in preprocessed ECG signal")

        # ---- handcrafted features ----
        extracted_features = extract_psd_features_dynamic(
            smoothed, fs=self.target_fs
        )
        features = np.concatenate(
            ([hr, rr_mean, rr_std, rmssd], extracted_features)
        )
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        if not torch.isfinite(feature_tensor).all():
            raise ValueError("NaNs/Infs in feature vector")

        # ---- DINO image ----
        dino_img = ecg_to_image_large_fix_nozscore_dino(
            smoothed_t, len(smoothed_t)
        )
        pixel_values = self._dino_transform(dino_img)

        return pixel_values.unsqueeze(0), feature_tensor.unsqueeze(0)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_model(checkpoint_path, device):
    config = ECGDinoConfig(
        image_model_name=IMAGE_MODEL_NAME,
        num_features=NUM_FEATURES,
        num_labels=NUM_LABELS,
    )
    base_model = ECGDinoModel(config)
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------
# One-time loading -- call this ONCE, reuse the result for every window
# --------------------------------------------------------------------------
def load_svt(checkpoint_path, device=None):
    """
    Builds everything infer_svt() needs (image processor, denoiser, model +
    LoRA adapter) exactly once. Pass the returned dict into infer_svt() as
    `resources` for every subsequent call instead of a checkpoint path --
    this is what avoids re-loading the model from disk on every window.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = ECGPreprocessPipeline()
    model = load_model(checkpoint_path, device)
    return {"model": model, "pipeline": pipeline, "device": device}


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def infer_svt(signal, fs, unit, checkpoint_path=None, device=None, pipeline=None,
              model=None, resources=None):
    """
    Run the SVT DINO model end-to-end on a single raw signal.

    Fast path (recommended for repeated / windowed calls):
        resources = load_svt(checkpoint_path)   # once
        infer_svt(window, fs, unit, resources=resources)   # per window

    Slow path (kept for backwards compatibility / one-off calls):
        infer_svt(signal, fs, unit, checkpoint_path)
        -- rebuilds the image processor + reloads the model from disk
           on every single call. Fine for a single ad-hoc inference,
           wasteful in a loop.

    Returns a dict:
        {"SVT": {"prob": float, "pred": int}}
    """
    if resources is not None:
        model = resources["model"]
        pipeline = resources["pipeline"]
        device = resources["device"]
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if pipeline is None:
            pipeline = ECGPreprocessPipeline()
        if model is None:
            if checkpoint_path is None:
                raise ValueError(
                    "infer_svt needs either `resources=load_svt(...)`, a pre-loaded "
                    "`model`, or a `checkpoint_path` to load one from."
                )
            model = load_model(checkpoint_path, device)

    pixel_values, features = pipeline(signal, fs, unit)
    pixel_values = pixel_values.to(device)
    features = features.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, ecg_features=features)
        logits = outputs["logits"].cpu().numpy()

    preds, probs = apply_alpha_threshold(logits, BEST_ALPHA, BEST_THRESHOLD)

    return {"SVT": {"prob": float(probs[0]), "pred": int(preds[0])}}


def main():
    parser = argparse.ArgumentParser(description="SVT DINO inference")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the PEFT/LoRA adapter dir for the SVT head"
    )
    # parser.add_argument(
    #     "--signal-npy", required=True, help="Path to a .npy file with the raw lead-II signal"
    # )
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz)")
    parser.add_argument("--unit", default="mV", help="Signal unit, e.g. mV")
    args = parser.parse_args()

    # signal = np.load(args.signal_npy)
    signal = np.random.randn(10000).astype(np.float32)
    result = infer_svt(signal, args.fs, args.unit, args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()