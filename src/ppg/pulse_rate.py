import numpy as np

def calculate_pulse_rate(peaks, fs):
    if len(peaks) < 2:
        return None

    peaks = np.array(peaks)

    pp_intervals_sec = np.diff(peaks) / fs
    pp_intervals_ms = pp_intervals_sec * 1000

    pr_bpm = 60 / pp_intervals_sec

    avg_pr = np.mean(pr_bpm)
    median_pr = np.median(pr_bpm)

    return pr_bpm, np.round(avg_pr, 2), np.round(median_pr, 2) 