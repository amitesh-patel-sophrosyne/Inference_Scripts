import neurokit2 as nk
import numpy as np

# ------------------------------------ Time domain features for Pulse Rate Variability --------------------------------

def calculate_pp_intervals(peaks, fs):
    if len(peaks) < 2:
        return [], 0, 0, 0

    pp_intervals = np.diff(peaks) / fs
    k = 0

    original_ppi = pp_intervals.copy()

    for start in range(0, len(pp_intervals)):
        if pp_intervals[start] >= 0.3 and pp_intervals[start] <= 1.5: # 1.8
            start_idx = start
            break

    for end in range(len(pp_intervals)-1, -1, -1):
        if pp_intervals[end] >= 0.3 and pp_intervals[end] <= 1.5:
            end_idx = end
            break

    for i in range(start_idx, end_idx+1):
        if pp_intervals[i] < 0.3 or pp_intervals[i] > 1.5:
            if pp_intervals[i+1] < 0.3 or pp_intervals[i+1] > 1.5:
                # find the next pp intervals which is lying between 0.3 and 1.8
                for j in range(i+2, len(pp_intervals)):
                    if pp_intervals[j] >= 0.3 and pp_intervals[j] <= 1.5:
                        pp_intervals[i] = np.mean([pp_intervals[i-1], pp_intervals[j]])
                        break
            else:
                pp_intervals[i] = np.mean([pp_intervals[i-1], pp_intervals[i+1]])
            k = k + 1
            
    return pp_intervals, np.min(pp_intervals), np.max(pp_intervals), np.mean(pp_intervals)

def calculate_sdnn(ppi, fs):
    #if len(ppi) < 2:
    #    return 0, 0
    
    pp_intervals_sec = ppi
    
    # Calculate SDNN (standard deviation of PP intervals)
    sdnn_sec = np.std(pp_intervals_sec, ddof=1) # ddof=1: n-1 for sample standard deviation

    sdnn_ms = sdnn_sec * 1000
    
    return sdnn_ms, sdnn_sec

def calculate_rmssd(ppi, fs):
    #if len(ppi) < 2:
    #    return 0, 0
    
    # convert pp intervals from sec to ms
    pp_intervals_ms = ppi * 1000

    diffs = np.diff(pp_intervals_ms)

    # square the differences
    diffs_squared = diffs ** 2

    # calculate RMSSD (root mean square of successive differences)
    rmssd_ms = np.sqrt(np.mean(diffs_squared))
    
    rmssd_sec = rmssd_ms / 1000
    
    return rmssd_ms, rmssd_sec


def calculate_nn50(ppi, fs):
    #if len(ppi) < 2:
    #    return 0, 0

    pp_intervals_sec = ppi

    # calculate NN50 (number of successive pairs of PP intervals that differ by more than 50ms)
    nn50 = np.sum(np.abs(np.diff(pp_intervals_sec)) > 0.05)

    pnn50 = nn50 / len(pp_intervals_sec) * 100
    
    return nn50, pnn50

def calculate_hrmax_hrmin(ppi, fs):

    pp_intervals_sec = ppi

    # calculate HRmax (maximum heart rate)
    hr = 60 /(pp_intervals_sec)

    # The average difference between the highest and lowest HRs during each respiratory cycle (HR Max − HR Min)
    hr_max_min = np.max(hr) - np.min(hr)

    return hr_max_min


def calculate_hrv_triangular_index(ppi, fs):
    pp_intervals_sec = ppi
    
    # calculate triangular index (number of PP intervals falling within each bin)
    hist, _ = np.histogram(pp_intervals_sec, bins='auto')
    triangular_index = len(pp_intervals_sec) / np.max(hist)
    
    return triangular_index


# ------------------------------------ Frequency domain features for Pulse Rate Variability --------------------------------

def compute_frequency_domain_hrv(peaks, fs):
    
    hrv_freq = nk.hrv_frequency(peaks, sampling_rate=fs)

    lf = hrv_freq['HRV_LF']
    hf = hrv_freq['HRV_HF']
    lf_hf_ratio = hrv_freq['HRV_LFHF']
    
    return lf, hf, lf_hf_ratio # millicond^2

# ------------------------------------ Non linear features for Pulse Rate Variability --------------------------------


def compute_nonlinear_hrv(ppi, fs):

    pp_intervals_sec = ppi

    if len(pp_intervals_sec) < 2:
        return None

    # Step 1: successive differences
    diff_pp = np.diff(pp_intervals_sec)

    # Step 2: variances
    var_pp = np.var(pp_intervals_sec, ddof=1)
    var_diff = np.var(diff_pp, ddof=1)

    # Step 3: SD1 and SD2
    sd1 = np.sqrt(0.5 * var_diff)
    sd2 = np.sqrt(2 * var_pp - 0.5 * var_diff)

    # Step 4: ratio
    if sd2 != 0:
        sd1_sd2_ratio = sd1 / sd2
    else:
        sd1_sd2_ratio = 0

    sd1, sd2 = sd1*1000, sd2*1000 # convert into ms

    return sd1, sd2, sd1_sd2_ratio
