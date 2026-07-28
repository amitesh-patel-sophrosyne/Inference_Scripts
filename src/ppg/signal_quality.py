# Signal quality check parameters
from scipy.signal import resample
import numpy as np
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# ----------------------------- SQI PARAMETERS -----------------------------------

def compute_perfusion_index(raw_ppg_signal, filtered_ppg_signal, valleys, peaks):

    raw_signal = np.array(raw_ppg_signal)
    filtered_signal = np.array(filtered_ppg_signal)
    pi_values = []

    num_beats = min(len(valleys), len(peaks))

    dc_values = []

    for i in range(1, num_beats-1):
        v1 = valleys[i]
        v2 = valleys[i + 1]
        p = peaks[i]

        if p > v1:
            ac = filtered_signal[p] - filtered_signal[v1]   # pulsatile component
            dc_signal = raw_signal[v1:v2]
            dc = np.mean(dc_signal)   # baseline component
            dc_values.append(dc)

            if dc == 0:
                pi_values.append(0)
                continue

            pi = (ac / abs(dc)) * 100
            pi_values.append(pi)

    pi_avg = np.mean(pi_values) if len(pi_values) > 0 else 0


    # ------------------ % Change ------------------
    pi_perc = []

    for i in range(len(pi_values) - 1):
        p1 = pi_values[i]
        p2 = pi_values[i + 1]

        if p1 != 0:
            perc = ((p2 - p1) / p1) * 100
            pi_perc.append(abs(perc))

    return pi_values, pi_avg, pi_perc, dc_values
    

def peak_to_peak_duration(peaks, fs):

    p2p_durations = []

    for i in range(1, len(peaks) - 1):
        p1 = peaks[i]
        p2 = peaks[i - 1]

        duration = (p1 - p2) / fs   # convert samples → seconds
        p2p_durations.append(duration)

    p2p_avg = np.mean(p2p_durations) if len(p2p_durations) > 0 else 0

    # ------------------ % Change ------------------
    p2p_perc = []

    for i in range(len(p2p_durations) - 1):
        d1 = p2p_durations[i]
        d2 = p2p_durations[i + 1]

        if d1 != 0:
            perc = ((d2 - d1) / d1) * 100
            p2p_perc.append(abs(perc))

    return p2p_durations, p2p_avg, p2p_perc



def compute_kurtosis(ppg_signal, valleys):

    ppg_signal = np.array(ppg_signal)
    kurtosis_values = []

    for i in range(1, len(valleys) - 1):
        segment = ppg_signal[valleys[i]:valleys[i+1]]

        if len(segment) < 2:
            continue

        mean = np.mean(segment)
        std = np.std(segment)

        if std == 0:
            kurtosis_values.append(0)
            continue

        N = len(segment)
        k = np.sum(((segment - mean) / std) ** 4) / N

        kurtosis_values.append(k)

    kur_avg = np.mean(kurtosis_values)
    # ------------------ % Change ------------------
    kur_perc = []

    for i in range(len(kurtosis_values) - 1):
        k1 = kurtosis_values[i]
        k2 = kurtosis_values[i + 1]

        if k1 != 0:
            perc = ((k2 - k1) / k1) * 100
            kur_perc.append(abs(perc))

    return kurtosis_values, kur_avg, kur_perc


def compute_entropy(ppg_signal, valleys):

    ppg_signal = np.array(ppg_signal)
    entropy_values = []

    for i in range(1, len(valleys) - 1):
        segment = ppg_signal[valleys[i]:valleys[i+1]]

        if len(segment) < 2:
            continue

        p = np.abs(segment)
        p = p / np.sum(p)
        entropy = -np.sum(p * np.log(p + 1e-12))

        entropy_values.append(entropy)

    ent_avg = np.mean(entropy_values)

    # ------------------ % Change ------------------
    ent_perc = []

    for i in range(len(entropy_values) - 1):
        e1 = entropy_values[i]
        e2 = entropy_values[i + 1]

        if e1 != 0:
            perc = ((e2 - e1) / e1) * 100
            ent_perc.append(abs(perc))

    return entropy_values, ent_avg, ent_perc


def compute_correlation(ppg_signal, valleys):
    
    signal = np.array(ppg_signal)

    rho_values = []

    for i in range(1, len(valleys) - 1):
        seg1 = signal[valleys[i-1]:valleys[i]]
        seg2 = signal[valleys[i]:valleys[i+1]]

        # resample to normalise length
        maxlen = max(len(seg1), len(seg2))
        seg1 = resample(seg1, maxlen)
        seg2 = resample(seg2, maxlen)

        mean1 = np.mean(seg1)
        mean2 = np.mean(seg2)

        std1 = np.std(seg1)
        std2 = np.std(seg2)

        if std1 == 0 or std2 == 0:
            rho_values.append(0)
            continue

        numerator = np.mean((seg1 - mean1) * (seg2 - mean2))
        rho = numerator / (std1 * std2)

        rho_values.append(rho)

    rho_avg = np.mean(rho_values)

    # ------------------ % Change ------------------
    rho_perc = []

    for i in range(len(rho_values) - 1):
        r1 = rho_values[i]
        r2 = rho_values[i + 1]

        if r1 != 0:
            perc = ((r2 - r1) / abs(r1)) * 100
            rho_perc.append(abs(perc))

    return rho_values, rho_avg, rho_perc


# calculate_fft & get_top_dominant_peaks used in calcualting the heart frequency for SNR-AC HF 
def calculate_fft(signal, fs):
    N = len(signal)
    
    X = fft(signal)
    f = fftfreq(N, 1/fs)

    # convert to amplitude
    X = (2 / N) * np.abs(X)
    
    return f, X



def get_top_dominant_peaks(f, X, top_n=3):
    """
    Get top N dominant PEAK frequencies (local maxima)

    Returns:
        list of tuples: [(freq, amplitude), ...]
    """
    # Only positive frequencies
    mask = f > 0
    f = f[mask]
    X = X[mask]

    # Find peaks
    peaks, _ = find_peaks(X)

    peak_freqs = f[peaks]
    peak_amps = X[peaks]

    # Sort peaks by amplitude
    sorted_idx = np.argsort(peak_amps)[-top_n:][::-1]

    top_peaks = [(peak_freqs[i], peak_amps[i]) for i in sorted_idx]

    return top_peaks



def compute_snr_ac_hf_2beats(raw_signal, filtered_signal, valleys, fs, hr_freq, num_harmonics=3):
    """
    Compute SNR-AC-HF using 2-beat windows

    Returns:
        list: SNR per 2-beat window
        float: average SNR
    """

    raw_signal = np.array(raw_signal)
    filtered_signal = np.array(filtered_signal)

    snr_values = []

    # Each window = 2 beats = valley[i] to valley[i+2]
    for i in range(len(valleys) - 2):
        start = valleys[i]
        end = valleys[i+2]

        raw_seg = raw_signal[start:end]
        filt_seg = filtered_signal[start:end]

        if len(filt_seg) < 10:
            continue

        # -------- SIGNAL POWER --------
        freqs_f, psd_f = welch(filt_seg, fs=fs)

        band_mask = (freqs_f >= 0.5) & (freqs_f <= 10)
        freqs_band = freqs_f[band_mask]
        psd_band = psd_f[band_mask]

        signal_mask = np.zeros_like(freqs_band, dtype=bool)

        for k in range(1, num_harmonics + 1):
            target_freq = k * np.float64(hr_freq)

            # Compute distance from target
            distances = np.abs(freqs_band - target_freq)

            # Sort indices by distance (ascending)
            sorted_idx = np.argsort(distances)

            if k == 1:
                n_bins = 5   
            else:
                n_bins = 3   

            selected_idx = sorted_idx[:n_bins]

            signal_mask[selected_idx] = True
        
        signal_power = np.sum(psd_band[signal_mask])

        # -------- NOISE POWER --------
        freqs_r, psd_r = welch(raw_seg, fs=fs)
        
        if fs >= 400:
            noise_mask = (freqs_r >= 150) & (freqs_r <= 190)
            noise_power = np.sum(psd_r[noise_mask]) / 4
        
        elif 200 <= fs < 400:
            noise_mask = (freqs_r >= 75) & (freqs_r <= 95)
            noise_power = np.sum(psd_r[noise_mask]) / 4
        
        elif 100 <= fs < 200:
            noise_mask = (freqs_r >= 40) & (freqs_r <= 48)
            noise_power = np.sum(psd_r[noise_mask]) / 4
        
        elif 60 <= fs < 100:
            noise_mask = (freqs_r >= 25) & (freqs_r <= 29)
            noise_power = np.sum(psd_r[noise_mask]) / 4
        
        else:
            noise_mask = (freqs_r >= 13) & (freqs_r <= 14)
            noise_power = np.sum(psd_r[noise_mask]) / 4


        # -------- SNR --------
        if noise_power <= 0 or signal_power <= 0:
            snr_values.append(0)
        else:
            snr = 10 * np.log10(signal_power / noise_power)
            snr_values.append(snr)

    snr_avg = np.mean(snr_values) if len(snr_values) > 0 else 0

    # ------------------ % Change ------------------
    snr_perc = []

    for i in range(len(snr_values) - 1):
        s1 = snr_values[i]
        s2 = snr_values[i + 1]

        if s1 != 0:
            perc = ((s2 - s1) / abs(s1)) * 100
            snr_perc.append(abs(perc)) 

    return snr_values, snr_avg, snr_perc


def compute_snr_ac_time_2beats(peaks, valleys, signal):
    signal = np.array(signal)

    num_beats = min(len(peaks), len(valleys))
    if num_beats < 2:
        return [], 0

    # -------- AC amplitudes --------
    ac_values = []
    for i in range(num_beats):
        ac = signal[peaks[i]] - signal[valleys[i]]
        ac_values.append(ac)

    ac_values = np.array(ac_values)

    # -------- SNR (2-beat window) --------
    snr_values = []

    for i in range(1, len(ac_values) - 1):   # 2-beat window
        window = ac_values[i:i+2]

        mean_ac = np.mean(window)
        std_ac = np.std(window)

        if std_ac == 0:
            snr = 0
        else:
            snr = 10 * np.log10((mean_ac / std_ac) ** 2)

        snr_values.append(snr)

    snr_avg = np.mean(snr_values) if len(snr_values) > 0 else 0
    # -------- % Change --------
    snr_perc = []

    for i in range(len(snr_values) - 1):
        s1 = snr_values[i]
        s2 = snr_values[i + 1]

        if s1 != 0:
            perc = ((s2 - s1) / abs(s1)) * 100
            snr_perc.append(abs(perc))

    return snr_values, snr_avg, snr_perc


# ------------------------------------- Signal Quality Check -------------------------------------

def signal_quality_check(activity, corr_thresh, pi_perc, kur_perc, entropy_perc, corr_perc, snrhf_perc, snrtime_perc, peak_peak_perc):
    beat_pos = []
    thresh_counter = []
    quality_class = []
    beat_thres = {}
    for i in range(len(corr_thresh)-1):
        beatnum = i+1
        cnt = 0
        beat_thres[beatnum] = {}

        if 'sit' in activity or 'rest' in activity: 
            if corr_thresh[i] > 0.7:
                cnt+=1
            else:
                beat_thres[beatnum]['corr_thresh'] = corr_thresh[i]
            
            if pi_perc[i] < 40:
                cnt+=1
            else:
                beat_thres[beatnum]['pi_perc'] = pi_perc[i]
                
            if kur_perc[i] < 40:
                cnt+=1
            else:
                beat_thres[beatnum]['kur_perc'] = kur_perc[i]

            if entropy_perc[i] < 8:
                cnt+=1
            else:
                beat_thres[beatnum]['entropy_perc'] = entropy_perc[i]
            
            if corr_perc[i] < 12:
                cnt+=1
            else:
                beat_thres[beatnum]['corr_perc'] = corr_perc[i]

            if snrhf_perc[i] < 12:
                cnt+=1
            else:
                beat_thres[beatnum]['snrhf_perc'] = snrhf_perc[i]

            if snrtime_perc[i] < 120:
                cnt+=1
            else:
                beat_thres[beatnum]['snrtime_perc'] = snrtime_perc[i]

            if peak_peak_perc[i] < 15:
                cnt+=1
            else:
                beat_thres[beatnum]['peak_peak_perc'] = peak_peak_perc[i]

        else:
            if corr_thresh[i] > 0.6:
                cnt+=1
            else:
                beat_thres[beatnum]['corr_thresh'] = corr_thresh[i]
            
            if pi_perc[i] < 60:
                cnt+=1
            else:
                beat_thres[beatnum]['pi_perc'] = pi_perc[i]
                
            if kur_perc[i] < 60:
                cnt+=1
            else:
                beat_thres[beatnum]['kur_perc'] = kur_perc[i]

            if entropy_perc[i] < 15:
                cnt+=1
            else:
                beat_thres[beatnum]['entropy_perc'] = entropy_perc[i]
            
            if corr_perc[i] < 16:
                cnt+=1
            else:
                beat_thres[beatnum]['corr_perc'] = corr_perc[i]

            if snrhf_perc[i] < 18:
                cnt+=1
            else:
                beat_thres[beatnum]['snrhf_perc'] = snrhf_perc[i]

            if snrtime_perc[i] < 150:
                cnt+=1
            else:
                beat_thres[beatnum]['snrtime_perc'] = snrtime_perc[i]

            if peak_peak_perc[i] < 20:
                cnt+=1
            else:
                beat_thres[beatnum]['peak_peak_perc'] = peak_peak_perc[i]


        if cnt >= 6:
            quality = 'good'
            beat_pos.append(beatnum)
            thresh_counter.append(cnt)
            quality_class.append(quality) 

        else:
            quality = 'bad'
            beat_pos.append(beatnum)
            thresh_counter.append(cnt)
            quality_class.append(quality) 

        good_cnt = 0
        for qc in quality_class:
            if qc == 'good':
                good_cnt += 1

        if (good_cnt/len(quality_class)) >= 0.7:
            overall_signal_quality = 'good'
        else:
            overall_signal_quality = 'bad'

    return beat_pos, thresh_counter, quality_class, beat_thres, overall_signal_quality


# to use this file:
# import signal_quality
# pi_values, pi_avg, pi_perc, dc_values = signal_quality.compute_perfusion_index(raw_ppg_signal, filtered_ppg_signal, valleys, peaks)
