import preprocessing as preprocessing
import peaks_valley_detection as peaks_valley_detection
import period_cycle_extraction as period_cycle_extraction
import signal_quality as signal_quality
import motion_artifacts_removal as motion_artifacts_removal
import spo2_calculation as spo2_calculation
import pulse_rate as pulse_rate
import pulse_rate_varibility as pulse_rate_varibility

def replace_missing_peaks_and_valleys(green_peaks, green_valleys, red_peaks, red_valleys, ir_peaks, ir_valleys, min_required=5):
    """
    Replace peaks/valleys of one channel using another channel
    if the current channel has insufficient detected peaks/valleys.

    Priority:
    - Green -> Red -> IR
    - Red   -> Green -> IR
    - IR    -> Green -> Red
    """

    # ---------------- Green ----------------
    if len(green_peaks) < min_required and len(green_valleys) < min_required:

        if len(red_peaks) >= min_required and len(red_valleys) >= min_required:
            green_peaks = red_peaks
            green_valleys = red_valleys

        elif len(ir_peaks) >= min_required and len(ir_valleys) >= min_required:
            green_peaks = ir_peaks
            green_valleys = ir_valleys

    # ---------------- Red ----------------
    if len(red_peaks) < min_required and len(red_valleys) < min_required:

        if len(green_peaks) >= min_required and len(green_valleys) >= min_required:
            red_peaks = green_peaks
            red_valleys = green_valleys

        elif len(ir_peaks) >= min_required and len(ir_valleys) >= min_required:
            red_peaks = ir_peaks
            red_valleys = ir_valleys

    # ---------------- IR ----------------
    if len(ir_peaks) < min_required and len(ir_valleys) < min_required:

        if len(green_peaks) >= min_required and len(green_valleys) >= min_required:
            ir_peaks = green_peaks
            ir_valleys = green_valleys

        elif len(red_peaks) >= min_required and len(red_valleys) >= min_required:
            ir_peaks = red_peaks
            ir_valleys = red_valleys

    return (green_peaks, green_valleys, red_peaks, red_valleys, ir_peaks, ir_valleys)

def ppg_preprocessing(green_signal, red_signal, ir_signal, avg_period, fs):
    """
    Preprocess PPG signals.
    """
    # baseline wander estimated signal 
    baseline_wander_green = preprocessing.moving_average_filter1(green_signal, fs=fs, window_sec=1.0)

    # baseline removed signal
    baseline_removed_sig_green = preprocessing.remove_baseline_wander1(green_signal, baseline_wander_green)
    
    # chebyshev type2 lowpass filter
    filtered_signal_green = preprocessing.chebyshev_type2_lowpass(baseline_removed_sig_green, fs=fs, cutoff=8, order=7, rs=40)
    
    # Red Preprocessing
    baseline_wander_red = preprocessing.moving_average_filter1(red_signal, fs=fs, window_sec=avg_period)
    baseline_removed_sig_red = preprocessing.remove_baseline_wander1(red_signal, baseline_wander_red)
    filtered_signal_red = preprocessing.chebyshev_type2_lowpass(baseline_removed_sig_red, fs=fs, cutoff=8, order=7, rs=40)
    

    # IR preprocessing 
    baseline_wander_ir = preprocessing.moving_average_filter1(ir_signal, fs=fs, window_sec=avg_period)
    baseline_removed_sig_ir = preprocessing.remove_baseline_wander1(ir_signal, baseline_wander_ir)
    filtered_signal_ir = preprocessing.chebyshev_type2_lowpass(baseline_removed_sig_ir, fs=fs, cutoff=8, order=7, rs=40)

    return filtered_signal_green, filtered_signal_red, filtered_signal_ir

def sqi_check(activity, peaks, valleys, raw_signal, filtered_signal, fs):
    """
    Check SQI of the signals.
    """
    # peak to peak duration
    _, _, peak_peak_perc = signal_quality.peak_to_peak_duration(peaks, fs)

    # perfusion index
    _, _, pi_perc, _ = signal_quality.compute_perfusion_index(raw_ppg_signal= raw_signal, filtered_ppg_signal= filtered_signal, valleys= valleys, peaks= peaks)

    # Kurtosis
    _, _, kurtosis_perc = signal_quality.compute_kurtosis(ppg_signal= filtered_signal, valleys= valleys)

    # Entropy
    _, _, entropy_perc = signal_quality.compute_entropy(ppg_signal= filtered_signal, valleys= valleys)
    
    # Correlation
    correlation, _, correlation_perc = signal_quality.compute_correlation(ppg_signal= filtered_signal, valleys= valleys)

    # SNR AC HF
    f, X = signal_quality.calculate_fft(signal=filtered_signal, fs=fs)
    freq_comp = signal_quality.get_top_dominant_peaks(f, X, top_n=1)

    _, _, snr_ac_hf_perc = signal_quality.compute_snr_ac_hf_2beats(raw_signal= raw_signal, filtered_signal= filtered_signal, valleys = valleys, fs= fs, hr_freq= freq_comp[0][0], num_harmonics=3)

    # SNR AC TIME
    _, _, snr_ac_time_perc = signal_quality.compute_snr_ac_time_2beats(peaks= peaks, valleys= valleys, signal= filtered_signal)

    # 5. Signal Quality Check : RED
    (beat_pos, 
    _, 
    beat_quality_class, 
    _, 
    overall_signal_quality) = signal_quality.signal_quality_check(activity = activity, 
                                                corr_thresh= correlation, 
                                                pi_perc= pi_perc, 
                                                kur_perc= kurtosis_perc, 
                                                entropy_perc = entropy_perc, 
                                                corr_perc= correlation_perc, 
                                                snrhf_perc= snr_ac_hf_perc, 
                                                snrtime_perc= snr_ac_time_perc, 
                                                peak_peak_perc= peak_peak_perc)

    return beat_pos, beat_quality_class, overall_signal_quality


def motion_artifact_removal(filtered_signal, fs):
    A1, D1, A2, D2, A3, D3, A4, D4, A5, D5, A6, D6, A6_1, reconstructed_signal = preprocessing.get_all_swt_levels(filtered_signal, fs=fs, level=6)
        
    _, _, _, _, _, _, _, _, _, _, _, _, _, reconstructed_signal = motion_artifacts_removal.trim_signals_to_length(filtered_signal, A1, D1, A2, D2, A3, D3, A4, D4, A5, D5, A6, D6, A6_1, reconstructed_signal)

    A1_2, D1_2, A2_2, D2_2, A3_2, D3_2, A4_2, D4_2, A5_2, D5_2, A6_2, D6_2, A6_1_2, reconstructed_signal_2 = preprocessing.get_all_swt_levels(reconstructed_signal, fs=fs, level=6)

    (A1_2, 
    D1_2, 
    A2_2, 
    D2_2, 
    A3_2, 
    D3_2, 
    A4_2, 
    D4_2, 
    A5_2, 
    D5_2, 
    A6_2, 
    D6_2, 
    A6_1_2, 
    reconstructed_signal_2) = motion_artifacts_removal.trim_signals_to_length(filtered_signal, 
                                                            A1_2, 
                                                            D1_2, 
                                                            A2_2, 
                                                            D2_2, 
                                                            A3_2, 
                                                            D3_2, 
                                                            A4_2, 
                                                            D4_2, 
                                                            A5_2, 
                                                            D5_2, 
                                                            A6_2, 
                                                            D6_2, 
                                                            A6_1_2, 
                                                            reconstructed_signal_2)
        
    A6_1_peaks, A6_1_valleys, _, _  = peaks_valley_detection.peak_valley_detection(A6_1_2, fs)

    upper_threshold, lower_threshold = motion_artifacts_removal.upper_lower_threshold(A6_1_2, A6_1_peaks, A6_1_valleys)
    
    A6_motion_artifact = motion_artifacts_removal.extract_motion_artifact_signal(A6_1_2, lower_threshold, upper_threshold)
    
    _, motion_artifact_removed = motion_artifacts_removal.remove_motion_artifact(filtered_signal=filtered_signal, 
                                                                            reconstructed_signal=reconstructed_signal, 
                                                                                approximation_coeffs=[A1_2, A2_2, A3_2, A4_2, A5_2], 
                                                                                detail_coeffs=[D1_2, D2_2, D3_2, D4_2, D5_2, D6_2], 
                                                                                motion_artifact_approximation=A6_motion_artifact, 
                                                                                wavelet='db4')
    return motion_artifact_removed

def full_preprocess(green_signal, red_signal, ir_signal, fs, activity='sit'):
    """
    Calculate SpO2 from green, red, and IR signals.
    """
    
    # 1. Initial Preprocessing to extract period cycle from green signal
    # baseline wander estimated signal 
    baseline_wander_green = preprocessing.moving_average_filter1(green_signal, fs=fs, window_sec=1.0)

    # baseline removed signal
    baseline_removed_sig_green = preprocessing.remove_baseline_wander1(green_signal, baseline_wander_green)

    # chebyshev type2 lowpass filter
    filtered_signal = preprocessing.chebyshev_type2_lowpass(baseline_removed_sig_green, fs=fs, cutoff=6, order=7, rs=40)


    # peaks and valleys of Green signal
    green_peaks, green_valleys, _, _ = peaks_valley_detection.peak_valley_detection(filtered_signal, fs)
    
    if len(green_peaks) == 0 or len(green_valleys) == 0:
        print("No peaks or valleys found in green signal")
        return (), (), (), (), (), ()

    # extract period cycles
    _, avg_period = period_cycle_extraction.extract_cycle_periods(green_valleys, fs)

    # 2. Secondary Preprocessing for all three signal

    _, filtered_signal_red, filtered_signal_ir = ppg_preprocessing(green_signal = green_signal, red_signal = red_signal, ir_signal = ir_signal, avg_period = avg_period, fs=fs)
    
    # 3. Peaks and Valleys Detection for all three signals
    red_peaks, red_valleys, _, _  = peaks_valley_detection.peak_valley_detection(filtered_signal_red, fs)
    ir_peaks, ir_valleys, _, _  = peaks_valley_detection.peak_valley_detection(filtered_signal_ir, fs)
    # Replace missing peaks and valleys with the most reliable signal peaks and valleys
    green_peaks, green_valleys, red_peaks, red_valleys, ir_peaks, ir_valleys = replace_missing_peaks_and_valleys(green_peaks, green_valleys, red_peaks, red_valleys, ir_peaks, ir_valleys)

    if len(red_peaks) == 0 or len(red_valleys) == 0 or len(ir_peaks) == 0 or len(ir_valleys) == 0:
        print("No peaks or valleys found in red or ir signal")
        return (), (), (), (), (), ()


    # 4. Signal Quality Check : RED
    (red_beat_pos, 
    red_quality_class, 
    overall_red_quality) = sqi_check(activity = activity, peaks = red_peaks, valleys = red_valleys, raw_signal = red_signal, filtered_signal = filtered_signal_red, fs = fs)

    # 6. if the signal is bad, pass it to motion artifacts removal and do the above peaks detection, sqi check again
    if overall_red_quality == 'bad':
        motion_artifact_removed_red = motion_artifact_removal(filtered_signal = filtered_signal_red, fs = fs)
        
        # peaks and valley detection on motion artifact removed signal
        rec_red_peaks, rec_red_valleys, _, _  = peaks_valley_detection.peak_valley_detection(motion_artifact_removed_red, fs)
        
        # SQI PARAMETERS ON MOTION ARTIFACT REMOVED SIGNAL
        red_rec_beat_pos, red_rec_quality_class, overall_red_rec_quality = sqi_check(activity = activity, peaks= rec_red_peaks, valleys = rec_red_valleys, raw_signal = red_signal, filtered_signal = motion_artifact_removed_red, fs = fs)

        red_sqi_beats_pos = red_rec_beat_pos
        red_beats_sqi_class = red_rec_quality_class
        final_red_signal = motion_artifact_removed_red
        final_red_peaks = rec_red_peaks
        final_red_valleys = rec_red_valleys

    else:
        overall_red_rec_quality = 'good' 
        red_sqi_beats_pos = red_beat_pos
        red_beats_sqi_class = red_quality_class
        final_red_signal = filtered_signal_red
        final_red_peaks = red_peaks
        final_red_valleys = red_valleys


    # 7. Signal Quality Check : IR
    ir_beat_pos, ir_quality_class, overall_ir_quality = sqi_check(activity = activity, peaks= ir_peaks, valleys = ir_valleys, raw_signal = ir_signal, filtered_signal = filtered_signal_ir, fs = fs)
       
    if len(ir_beat_pos) == 0 or len(ir_quality_class) == 0:
        print("No beats or quality class found in ir signal")
        return (), (), (), (), (), ()

    if overall_ir_quality == 'bad':
        motion_artifact_removed_ir = motion_artifact_removal(filtered_signal = filtered_signal_ir, fs = fs)
        
        # peaks and valley detection on motion artifact removed signal
        rec_ir_peaks, rec_ir_valleys, _, _  = peaks_valley_detection.peak_valley_detection(motion_artifact_removed_ir, fs)
        
        # SQI Parameters of removed motion artifact IR signal 
        ir_rec_beat_pos, ir_rec_quality_class, overall_ir_rec_quality = sqi_check(activity = activity, peaks= rec_ir_peaks, valleys = rec_ir_valleys, raw_signal = ir_signal, filtered_signal = motion_artifact_removed_ir, fs = fs)
        
        if (len(ir_rec_beat_pos) == 0 or len(ir_rec_quality_class) == 0):
            print("No beats or quality class found in ir signal after motion artifact removal")
            return (), (), (), (), (), ()


        ir_sqi_beats_pos = ir_rec_beat_pos
        ir_beats_sqi_class = ir_rec_quality_class
        final_ir_signal = motion_artifact_removed_ir
        final_ir_peaks = rec_ir_peaks
        final_ir_valleys = rec_ir_valleys
    
    else: 
        overall_ir_rec_quality = 'good'
        ir_sqi_beats_pos = ir_beat_pos
        ir_beats_sqi_class = ir_quality_class
        final_ir_signal = filtered_signal_ir
        final_ir_peaks = ir_peaks
        final_ir_valleys = ir_valleys


    return (final_red_signal, final_ir_signal), (final_red_peaks, final_ir_peaks), (final_red_valleys, final_ir_valleys), (overall_red_rec_quality, overall_ir_rec_quality), (red_sqi_beats_pos, ir_sqi_beats_pos), (red_beats_sqi_class, ir_beats_sqi_class)


def calculate_spo2(raw_red_signal, raw_ir_signal, final_red_signal, final_ir_signal, red_sqi_beats_pos, ir_sqi_beats_pos, red_sqi_class, ir_sqi_class, red_peaks, ir_peaks, red_valleys, ir_valleys, fs):
    # 8. spo2 calculation
    aligned_red_beats_idx, aligned_ir_beats_idx, aligned_red_valleys1_idx, aligned_red_vallleys2_idx, aligned_ir_valleys1_idx, aligned_ir_valleys2_idx, aligned_red_classes, aligned_ir_classes = spo2_calculation.find_align_beats(red_sqi_beats_pos=red_sqi_beats_pos, ir_sqi_beats_pos=ir_sqi_beats_pos, red_sqi_class=red_sqi_class, ir_sqi_class=ir_sqi_class, red_peaks=red_peaks, ir_peaks=ir_peaks, red_valleys=red_valleys, ir_valleys=ir_valleys, fs=fs)
    mask_indices, aligned_red_beats_idx, aligned_ir_beats_idx = spo2_calculation.find_good_beats_pair(aligned_red_beats_idx=aligned_red_beats_idx, aligned_ir_beats_idx=aligned_ir_beats_idx, aligned_red_classes=aligned_red_classes, aligned_ir_classes=aligned_ir_classes, fs=fs)
    good_red_beats, good_ir_beats, good_red_valley_beats, good_ir_valley_beats = spo2_calculation.extract_good_beats(mask_indices, aligned_red_beats_idx, aligned_ir_beats_idx, aligned_red_valleys1_idx, aligned_red_vallleys2_idx, aligned_ir_valleys1_idx, aligned_ir_valleys2_idx)


    red_ac_values = spo2_calculation.calculate_AC(filtered_signal=final_red_signal, peaks=good_red_beats, valleys=good_red_valley_beats)
    red_dc_values = spo2_calculation.calculate_DC(raw_signal=raw_red_signal, valleys=good_red_valley_beats)
    ir_ac_values = spo2_calculation.calculate_AC(filtered_signal=final_ir_signal, peaks=good_ir_beats, valleys=good_ir_valley_beats)
    ir_dc_values = spo2_calculation.calculate_DC(raw_signal=raw_ir_signal, valleys=good_ir_valley_beats)
    r_values = spo2_calculation.calculate_R(ac_red=red_ac_values, ac_ir=ir_ac_values, dc_red=red_dc_values, dc_ir=ir_dc_values)
    spo2_values_linear, spo2_values_quadratic, spo2_avg_linear, spo2_avg_quadratic, spo2_median_linear, spo2_median_quadratic = spo2_calculation.spo2_calculator(r_values)
    #print(f'Measured spo2 values linear: {spo2_values_linear}, Measured spo2 values quadratic: {spo2_values_quadratic}')
    #print(f'Measured spo2 avg linear: {spo2_avg_linear}, Measured spo2 avg quadratic: {spo2_avg_quadratic}')
    #print(f'Measured spo2 median linear: {spo2_median_linear}, Measured spo2 median quadratic: {spo2_median_quadratic}')
    #print(f'Reference start spo2: {start_spo2}')
    #print(f'Reference end spo2: {end_spo2}')

    return spo2_values_linear, spo2_values_quadratic, spo2_avg_linear, spo2_avg_quadratic, spo2_median_linear, spo2_median_quadratic

