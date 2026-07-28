import os
import spo2_pipeline
# from pulse_rate import calculate_pulse_rate 
import pulse_rate 
import  pulse_rate_varibility

def process_signal(green_signal, red_signal, ir_signal, fs, activity='sit'):

    # inverted signals
    inverted_green = -green_signal
    inverted_red = -red_signal
    inverted_ir = -ir_signal

    (final_red_signal, final_ir_signal), (final_red_peaks, final_ir_peaks), (final_red_valleys, final_ir_valleys), (overall_red_rec_quality, overall_ir_rec_quality), (red_sqi_beats_pos, ir_sqi_beats_pos), (red_beats_sqi_class, ir_beats_sqi_class) = spo2_pipeline.full_preprocess(green_signal = inverted_green, red_signal = inverted_red, ir_signal = inverted_ir, fs = fs, activity=activity)
    spo2_values_linear, spo2_values_quadratic, spo2_avg_linear, spo2_avg_quadratic, spo2_median_linear, spo2_median_quadratic = spo2_pipeline.calculate_spo2(raw_red_signal=inverted_red, raw_ir_signal=inverted_ir, final_red_signal=final_red_signal, final_ir_signal=final_ir_signal, red_sqi_beats_pos=red_sqi_beats_pos, ir_sqi_beats_pos=ir_sqi_beats_pos, red_sqi_class=red_beats_sqi_class, ir_sqi_class=ir_beats_sqi_class, red_peaks=final_red_peaks, ir_peaks=final_ir_peaks, red_valleys=final_red_valleys, ir_valleys=final_ir_valleys, fs=fs)

    red_pr_bpm, red_avg_pr, red_median_pr = pulse_rate.calculate_pulse_rate(final_red_peaks, fs)
    ir_pr_bpm, ir_avg_pr, ir_median_pr = pulse_rate.calculate_pulse_rate(final_ir_peaks, fs)

    red_pp_intervals, red_min_pp_intervals, red_max_pp_intervals, red_mean_pp_intervals = pulse_rate_varibility.calculate_pp_intervals(final_red_peaks, fs)
    ir_pp_intervals, ir_min_pp_intervals, ir_max_pp_intervals, ir_mean_pp_intervals = pulse_rate_varibility.calculate_pp_intervals(final_ir_peaks, fs)

    red_sdnn_ms, red_sdnn_sec = pulse_rate_varibility.calculate_sdnn(red_pp_intervals, fs)
    ir_sdnn_ms, ir_sdnn_sec = pulse_rate_varibility.calculate_sdnn(ir_pp_intervals, fs)
    
    red_rmssd_ms, red_rmssd_sec = pulse_rate_varibility.calculate_rmssd(red_pp_intervals, fs)
    ir_rmssd_ms, ir_rmssd_sec = pulse_rate_varibility.calculate_rmssd(ir_pp_intervals, fs)
    
    red_nn50, red_pnn50 = pulse_rate_varibility.calculate_nn50(red_pp_intervals, fs)
    ir_nn50, ir_pnn50 = pulse_rate_varibility.calculate_nn50(ir_pp_intervals, fs)
    
    red_hr_max_min = pulse_rate_varibility.calculate_hrmax_hrmin(red_pp_intervals, fs)
    ir_hr_max_min = pulse_rate_varibility.calculate_hrmax_hrmin(ir_pp_intervals, fs)

    red_triangular_index = pulse_rate_varibility.calculate_hrv_triangular_index(red_pp_intervals, fs)
    ir_triangular_index = pulse_rate_varibility.calculate_hrv_triangular_index(ir_pp_intervals, fs)

    red_lf, red_hf, red_lf_hf_ratio = pulse_rate_varibility.compute_frequency_domain_hrv(final_red_peaks, fs)
    ir_lf, ir_hf, ir_lf_hf_ratio = pulse_rate_varibility.compute_frequency_domain_hrv(final_ir_peaks, fs)
    
    red_sd1, red_sd2, red_sd1_sd2_ratio = pulse_rate_varibility.compute_nonlinear_hrv(red_pp_intervals, fs)
    ir_sd1, ir_sd2, ir_sd1_sd2_ratio = pulse_rate_varibility.compute_nonlinear_hrv(ir_pp_intervals, fs)

    if overall_red_rec_quality == 'good' and overall_ir_rec_quality == 'good':
        print('RED: GOOD QUALITY, IR: GOOD QUALITY')
    elif overall_red_rec_quality == 'good' and overall_ir_rec_quality == 'bad':
        print('RED: GOOD QUALITY, IR: BAD QUALITY')
    elif overall_red_rec_quality == 'bad' and overall_ir_rec_quality == 'good':
        print('RED: BAD QUALITY, IR: GOOD QUALITY')
    elif overall_red_rec_quality == 'bad' and overall_ir_rec_quality == 'bad':
        print('RED: BAD QUALITY, IR: BAD QUALITY')

    # convert ppi from sec to ms
    red_pp_intervals_ms = red_pp_intervals * 1000
    ir_pp_intervals_ms = ir_pp_intervals * 1000
    red_max_pp_intervals_ms = red_max_pp_intervals * 1000
    ir_max_pp_intervals_ms = ir_max_pp_intervals * 1000
    red_min_pp_intervals_ms = red_min_pp_intervals * 1000
    ir_min_pp_intervals_ms = ir_min_pp_intervals * 1000
    red_mean_pp_intervals_ms = red_mean_pp_intervals * 1000
    ir_mean_pp_intervals_ms = ir_mean_pp_intervals * 1000

    filtered_signals = {
        'red_signal': final_red_signal,
        'ir_signal': final_ir_signal
    }
    signal_quality = {
        'red_signal_quality': overall_red_rec_quality,
        'ir_signal_quality': overall_ir_rec_quality
    }

    peaks = {
        'red_peaks': final_red_peaks,
        'red_valleys': final_red_valleys,
        'ir_peaks': final_ir_peaks,
        'ir_valleys': final_ir_valleys
    }

    spo2 = {
        'spo2_values_linear': spo2_values_linear,
        'spo2_values_quadratic': spo2_values_quadratic,
        'spo2_avg_linear': spo2_avg_linear,
        'spo2_avg_quadratic': spo2_avg_quadratic,
        'spo2_median_linear': spo2_median_linear,
        'spo2_median_quadratic': spo2_median_quadratic
    }

    Pulse_Rate = {
        'red_pulse_rate_bpm': red_pr_bpm,
        'red_pulse_rate_avg': red_avg_pr,
        'red_pulse_rate_median': red_median_pr,
        'ir_pulse_rate_bpm': ir_pr_bpm,
        'ir_pulse_rate_avg': ir_avg_pr,
        'ir_pulse_rate_median': ir_median_pr
    }

    pulse_rate_variability = {
        'red_pp_intervals_ms': red_pp_intervals_ms,
        'ir_pp_intervals_ms': ir_pp_intervals_ms,
        'red_max_pp_intervals_ms': red_max_pp_intervals_ms,
        'ir_max_pp_intervals_ms': ir_max_pp_intervals_ms,
        'red_min_pp_intervals_ms': red_min_pp_intervals_ms,
        'ir_min_pp_intervals_ms': ir_min_pp_intervals_ms,
        'red_mean_pp_intervals_ms': red_mean_pp_intervals_ms,
        'ir_mean_pp_intervals_ms': ir_mean_pp_intervals_ms,
        'red_rmssd': red_rmssd_ms,
        'ir_rmssd': ir_rmssd_ms,
        'red_sdnn': red_sdnn_ms,
        'ir_sdnn': ir_sdnn_ms,
        'red_pnn50': red_pnn50,
        'ir_pnn50': ir_pnn50,
        'red_sd1': red_sd1,
        'ir_sd1': ir_sd1,
        'red_sd2': red_sd2,
        'ir_sd2': ir_sd2,
        'red_sd1_sd2_ratio': red_sd1_sd2_ratio,
        'ir_sd1_sd2_ratio': ir_sd1_sd2_ratio,
        'red_lf': red_lf,
        'ir_lf': ir_lf,
        'red_hf': red_hf,
        'ir_hf': ir_hf,
        'red_lf_hf_ratio': red_lf_hf_ratio,
        'ir_lf_hf_ratio': ir_lf_hf_ratio,
        'red_triangular_index': red_triangular_index,
        'ir_triangular_index': ir_triangular_index
    }

    return {
        'filtered_signals': filtered_signals,
        'peaks': peaks,
        'signal_quality': signal_quality,
        'spo2': spo2, 
        'pulse_rate': Pulse_Rate, 
        'pulse_rate_variability': pulse_rate_variability
    }


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import traceback

    # -----------------------------------
    # Load CSV
    # -----------------------------------
    df = pd.read_csv(
        "/Users/amiteshpatel/Desktop/Sophro/IMU_Models/ppg_model/real_data/Deepak_Volts_20260707_155335.csv"
    )

    fs = 500

    ir_signal = df["Ch : LED 1 (IR) - LED 1 (IR) AMBIENT"].astype(float).to_numpy()
    red_signal = df["Ch : LED 2 (RED) - LED 2 (RED) AMBIENT"].astype(float).to_numpy()

    # If you don't have a green channel, duplicate IR
    green_signal = ir_signal.copy()

    print(f"Number of samples : {len(red_signal)}")
    print(f"Duration          : {len(red_signal)/fs:.2f} sec")

    # -----------------------------------
    # Optional: Use first 60 seconds
    # -----------------------------------
    duration = 60  # seconds
    samples = min(len(red_signal), duration * fs)

    green_signal = green_signal[:samples]
    red_signal = red_signal[:samples]
    ir_signal = ir_signal[:samples]

    try:

        results = process_signal(
            green_signal=green_signal,
            red_signal=red_signal,
            ir_signal=ir_signal,
            fs=fs,
            activity="sit",
        )

        print("\n==========================")
        print("Processing Successful")
        print("==========================")

        print("\nSignal Quality")
        print(results["signal_quality"])

        print("\nSpO₂")
        for k, v in results["spo2"].items():
            print(f"{k:30}: {v}")

        print("\nPulse Rate")
        for k, v in results["pulse_rate"].items():
            print(f"{k:30}: {v}")

        print("\nHRV")
        for k, v in results["pulse_rate_variability"].items():
            print(f"{k:30}: {v}")

        # -----------------------------------
        # Plot Filtered Signals (2 rows, 1 column)
        # -----------------------------------

        filtered_red = results["filtered_signals"]["red_signal"]
        filtered_ir = results["filtered_signals"]["ir_signal"]

        red_peaks = results["peaks"]["red_peaks"]
        ir_peaks = results["peaks"]["ir_peaks"]

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        # ---------------- RED ----------------
        axes[0].plot(filtered_red, color="red", label="Filtered Red")

        if len(red_peaks):
            axes[0].scatter(
                red_peaks,
                filtered_red[red_peaks],
                color="black",
                s=20,
                label="Peaks",
                zorder=3,
            )

        axes[0].set_title("Filtered RED Signal")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)
        axes[0].legend()

        # ---------------- IR ----------------
        axes[1].plot(filtered_ir, color="blue", label="Filtered IR")

        if len(ir_peaks):
            axes[1].scatter(
                ir_peaks,
                filtered_ir[ir_peaks],
                color="red",
                s=20,
                label="Peaks",
                zorder=3,
            )

        axes[1].set_title("Filtered IR Signal")
        axes[1].set_xlabel("Samples")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    except Exception:

        print("\nPipeline crashed.\n")
        traceback.print_exc()