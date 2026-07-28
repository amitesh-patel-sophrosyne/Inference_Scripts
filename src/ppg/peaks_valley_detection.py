import numpy as np

def peak_valley_detection(filtered_ppg, fs):
    '''Part-1 : Incremental merge segmentation'''
    seg = 0
    z = 0
    seginline = 1
    m = 1 #max(1, fs // fs) #1
    r = 0
    i = 1

    start_point_line  = np.zeros_like(filtered_ppg, dtype=float)
    start_index_line  = np.zeros_like(filtered_ppg, dtype=int)
    end_point_line    = np.zeros_like(filtered_ppg, dtype=float)
    end_index_line    = np.zeros_like(filtered_ppg, dtype=int)
    slope_line        = np.zeros_like(filtered_ppg, dtype=float)
    slope_seg         = np.zeros_like(filtered_ppg, dtype=float)

    # first segment
    start_point_line[z]  = filtered_ppg[seg]
    start_index_line[z]  = seg
    end_point_line[z]    = filtered_ppg[seg+m]
    end_index_line[z]    = seg+m
    slope_line[z]        = (end_point_line[z] - start_point_line[z]) / m
    slope_seg[z]         = slope_line[z]
    z += 1
    seg += 1

    # build all lines
    while (seg+1)*m < len(filtered_ppg)-2:
        slope_seg[i] = (filtered_ppg[(seg+1)*m] - filtered_ppg[seg*m]) / m
        if np.sign(slope_seg[i]) == np.sign(slope_seg[i-1]):
            slope_seg[i] = (filtered_ppg[(seg+1)*m] - filtered_ppg[(seg-seginline)*m]) / ((seg+1-seginline)*m)
            r = seginline
            seginline += 1
            seg += 1
        else:
            if seg > 1 and r >= 1:
                start_point_line[z]  = filtered_ppg[(seg-r-1)*m]
                end_point_line[z]    = filtered_ppg[seg*m]
                start_index_line[z]  = (seg-1-r)*m
                end_index_line[z]    = seg*m
                slope_line[z]        = (end_point_line[z] - start_point_line[z]) / ((end_index_line[z] - start_index_line[z]))
                r = 0
            z += 1
            seg += 1
            seginline = 1
        i += 1

    # trim to actual size
    start_point_line = start_point_line[:z]
    end_point_line   = end_point_line[:z]
    start_index_line = start_index_line[:z]
    end_index_line   = end_index_line[:z]
    slope_line       = slope_line[:z]

    '''Part-2: Adaptive Thresholding for peaks'''
    line_amplitude = np.abs(start_point_line - end_point_line)

    fidx  = np.where(start_index_line <= 2*fs)[0]                          # NEW: 2 s window
    diffs = filtered_ppg[start_index_line[fidx]] - filtered_ppg[end_index_line[fidx]]
    top5_sorted  = np.argsort(diffs)[-5:]                                   # indices of top-5
    top5_idx     = fidx[top5_sorted]                                        # global line indices
    first_peak_index = top5_idx[np.argmax(diffs[top5_sorted])]             # kept for downstream
    anchor_amp   = np.median(line_amplitude[top5_idx])                     # robust median anchor
    threshold_low = max(0.4 * anchor_amp, 0.2)                             # NEW: median-based
    global_peak_amplitude = np.percentile(filtered_ppg, 90) - np.percentile(filtered_ppg, 5)

    peak_index = np.zeros_like(start_index_line, dtype=int)
    peak_point = np.zeros_like(start_point_line, dtype=float)
    idx = 1
    flag_miss_count = 0

    j = first_peak_index
    while j < len(line_amplitude):
        if slope_line[j] < 0 and slope_line[j-1] > 0:#slope_line[j] < 0 and slope_line[j-1] != 0:
            # --- FIX: Measure prominence against 0.5s local minimum ---
            # Using line_amplitude[j] fails if the downstroke is split by a notch.
            # Instead, we measure the peak's height relative to the recent 0.5s baseline.
            pk_idx = start_index_line[j]
            pk_val = start_point_line[j]
            search_start = max(0, pk_idx - int(0.8 * fs))
            # if search_start is valid, find local min, otherwise use pk_val to avoid error
            local_min = np.min(filtered_ppg[search_start : pk_idx]) if pk_idx > search_start else 0
            prominence = pk_val - local_min

            adaptive_thr = max(0.10 * global_peak_amplitude, min(threshold_low, 0.35 * global_peak_amplitude))

            if prominence >= adaptive_thr:
                peak_index[idx] = pk_idx
                peak_point[idx] = pk_val
                idx += 1
                threshold_low = (threshold_low + 0.3 * prominence) / 2
                flag_miss_count = 0
            else:
                flag_miss_count += 1
                threshold_low = (threshold_low + prominence * max(0.2 - 0.1*flag_miss_count, 0)) / 2
        j += 1

    # keep only detected peaks
    peak_index = peak_index[1:idx]
    peak_point = peak_point[1:idx]

    # --- Improvement 7.3: Minimum inter-peak distance filter ---
    # Fix: any two peaks spaced < 0.3 s are almost certainly a double-detection.
    #      Keep the higher-amplitude one, drop the other.
    # Threshold: 0.3 s ≈ 200 BPM (maximum physiological heart rate).
    if len(peak_index) > 1:
        min_pk_gap = int(0.3 * fs)          # samples equivalent to ~200 BPM
        keep_mask  = np.ones(len(peak_index), dtype=bool)
        for jj in range(1, len(peak_index)):
            if keep_mask[jj-1] and (peak_index[jj] - peak_index[jj-1]) < min_pk_gap:
                # drop whichever of the pair has the lower amplitude
                if peak_point[jj] >= peak_point[jj-1]:
                    keep_mask[jj-1] = False
                else:
                    keep_mask[jj]   = False
        peak_index = peak_index[keep_mask]
        peak_point = peak_point[keep_mask]

# --- valley detection with distance + slope+monotonicity check ---
    if len(peak_index) > 1:
        median_pk_dist  = np.median(np.diff(peak_index))
        min_valley_dist = 0.25 * median_pk_dist
    else:
        min_valley_dist = 0

    valley_index = []
    N = len(filtered_ppg)
    for k, p in enumerate(peak_index):
        start = peak_index[k-1] if k>0 else 0
        end   = p

        # candidate with distance
        if valley_index:
            cand = int(valley_index[-1] + min_valley_dist)
            # clamp so start < end-1
            start = min(cand, end-2)
        
        # clamp start to valid range
        start = max(0, min(start, N-2))
        end   = max(start+1, min(end, N-1))

        seg = filtered_ppg[start:end]
        if seg.size < 2:
            v = start
        else:
            # find all local minima
            rel_mins = np.where((np.r_[True, seg[1:]<=seg[:-1]] &
                                 np.r_[seg[:-1]<=seg[1:], True]))[0]
            rel_mins = rel_mins[np.argsort(seg[rel_mins])]

            for m_rel in rel_mins:
                v_cand = start + m_rel
                # --- Improvement 7.4: Relaxed monotonicity check (80% rule) ---
                # NEW: accept if ≥ 80% of upstroke steps are rising — tolerates noise.
                upstroke = np.diff(filtered_ppg[v_cand:p])
                if upstroke.size > 0 and np.mean(upstroke >= 0) >= 0.80:  # NEW: 80% rule
                    v = v_cand
                    break
            else:
                # fallback to absolute minimum
                m_rel = np.argmin(seg)
                v = start + m_rel

        valley_index.append(int(v))
    
    

    valley_index = np.array(valley_index, dtype=int)
    valley_point = filtered_ppg[valley_index]
    
    # --- 3) sanity-check onset→peak amplitude & local slope ---
    # Calculate median amplitude of current candidates for dynamic thresholding
    median_amp = np.median(peak_point - valley_point) if len(peak_point) > 0 else 0.2
    
    grad = np.gradient(filtered_ppg)
    win  = int(0.06 * fs)  # 20 ms window for pre/post slope check
    
    keep = []
    N = len(filtered_ppg)
    for k, (pk, v) in enumerate(zip(peak_index, valley_index)):
        # 1) Amplitude drop from previous valley (Left drop)
        drop_left = peak_point[k] - valley_point[k]
        
        # 2) Amplitude drop to next valley (Right drop)
        if k < len(peak_index) - 1:
            next_pk = peak_index[k+1]
            drop_right = peak_point[k] - np.min(filtered_ppg[pk:next_pk])
        else:
            # For the last peak, just look forward 0.5s or to the end
            drop_right = peak_point[k] - np.min(filtered_ppg[pk : min(N, pk + int(0.5*fs))])
            
        # --- Median Amplitude Thresholds ---
        # The peak must rise significantly above its preceding valley (Reject diastolic & squiggles)
        if drop_left < 0.30 * median_amp:
            continue

        if drop_right < 0.25 * median_amp:
            continue
    
        # 2) local rise then fall
        pre_slope  = np.mean( grad[max(0, v):pk] )
        post_slope = np.mean( grad[ pk : min(len(filtered_ppg), pk+win) ] )
        if pre_slope <= 0:
            continue

        if post_slope > 0.05 * abs(pre_slope):
            continue
    
        keep.append(k)
    
    # re-index
    keep = np.array(keep, dtype=int)
    peak_index   = peak_index[keep]
    peak_point   = peak_point[keep]
    valley_index = valley_index[keep]
    valley_point = valley_point[keep]

    # --- Improvement 7.5 (FIXED): Inter-Beat Interval (IBI) consistency filter ---
    # NEW FIX: Only target stray peaks creating suspiciously SHORT gaps (< 70% of median).
    if len(peak_index) >= 3:
        ibi        = np.diff(peak_index)
        median_ibi = np.median(ibi)
        min_allowed_ibi = 0.70 * median_ibi  # Any gap < 70% of normal is a stray detection
        
        ibi_keep   = np.ones(len(peak_index), dtype=bool)
        for jj in range(1, len(peak_index)):
            if (peak_index[jj] - peak_index[jj-1]) < min_allowed_ibi:
                # Sandwiched stray detection found. Drop the lower amplitude one.
                if peak_point[jj] >= peak_point[jj-1]:
                    ibi_keep[jj-1] = False
                else:
                    ibi_keep[jj] = False
        peak_index   = peak_index[ibi_keep]
        peak_point   = peak_point[ibi_keep]
        valley_index = valley_index[ibi_keep]
        valley_point = valley_point[ibi_keep]

    return peak_index, valley_index, peak_point, valley_point