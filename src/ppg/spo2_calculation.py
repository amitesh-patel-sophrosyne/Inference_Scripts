
import numpy as np

# -------------------------------------------- Find aligned beats- RED & IR ---------------------------------
def find_align_beats(red_sqi_beats_pos, ir_sqi_beats_pos, red_sqi_class, ir_sqi_class, red_peaks, ir_peaks, red_valleys, ir_valleys, fs):
    """
    Find aligned beats between RED and IR signals.
    
    Returns
    -------
    aligned_red_beats_idx : list
        Aligned RED beat indices.
    aligned_ir_beats_idx : list
        Aligned IR beat indices.
    aligned_red_valleys1_idx : list
        Aligned RED valley 1 indices.
    aligned_red_vallleys2_idx : list
        Aligned RED valley 2 indices.
    aligned_ir_valleys1_idx : list
        Aligned IR valley 1 indices.
    aligned_ir_valleys2_idx : list
        Aligned IR valley 2 indices.
    aligned_red_classes : list
        Aligned RED beat classes.
    aligned_ir_classes : list
        Aligned IR beat classes.
    """
    min_len = min(len(red_sqi_beats_pos), len(ir_sqi_beats_pos))

    aligned_red_beats_idx = []
    aligned_ir_beats_idx = []
    aligned_red_valleys1_idx = []
    aligned_red_vallleys2_idx = []
    aligned_ir_valleys1_idx = []
    aligned_ir_valleys2_idx = []
    aligned_red_classes = []
    aligned_ir_classes = []
    
    for i in range(min_len):
        red_pos = red_sqi_beats_pos[i]
        
        red_peak_idx = red_peaks[red_pos]
        red_beat_time = red_peak_idx/fs

        for j in range(min_len):
            ir_pos = ir_sqi_beats_pos[j]
            ir_peak_idx = ir_peaks[ir_pos]
            ir_beat_time = ir_peak_idx/fs
            #if red_beat_time == ir_beat_time:
            if (red_beat_time - 0.05) <= ir_beat_time <= (red_beat_time + 0.05):
                aligned_red_beats_idx.append(red_peak_idx)
                aligned_ir_beats_idx.append(ir_peak_idx)
                aligned_red_valleys1_idx.append(red_valleys[red_pos])
                aligned_red_vallleys2_idx.append(red_valleys[red_pos+1])
                aligned_ir_valleys1_idx.append(ir_valleys[ir_pos])
                aligned_ir_valleys2_idx.append(ir_valleys[ir_pos+1])

                aligned_red_classes.append(red_sqi_class[i])
                aligned_ir_classes.append(ir_sqi_class[j])
                break
    return aligned_red_beats_idx, aligned_ir_beats_idx, aligned_red_valleys1_idx, aligned_red_vallleys2_idx, aligned_ir_valleys1_idx, aligned_ir_valleys2_idx, aligned_red_classes, aligned_ir_classes



# -------------------------------------------- Find good beats pair ---------------------------------
def find_good_beats_pair(aligned_red_beats_idx, aligned_ir_beats_idx, aligned_red_classes, aligned_ir_classes, fs):
    """
    Find good beats pair.
    
    Returns
    -------
    mask_indices : list
        Mask indices.
    aligned_red_beats_idx : list
        Aligned RED beat indices.
    aligned_ir_beats_idx : list
        Aligned IR beat indices.
    """
    mask_indices = [0] * len(aligned_red_beats_idx)
    for i in range(len(aligned_red_beats_idx)):
        if aligned_red_classes[i] == 'good' and aligned_ir_classes[i] == 'good':
            mask_indices[i] = 1
    return mask_indices, aligned_red_beats_idx, aligned_ir_beats_idx


def extract_good_beats(
    mask_indices,
    aligned_red_beats_idx,
    aligned_ir_beats_idx,
    aligned_red_valleys1_idx,
    aligned_red_vallleys2_idx,
    aligned_ir_valleys1_idx,
    aligned_ir_valleys2_idx
):
    """
    Extract valid beats and valleys based on mask indices.

    Parameters
    ----------
    mask_indices : list or array
        Mask values (1 = good beat).

    aligned_red_beats_idx : list
        Red peak indices.

    aligned_ir_beats_idx : list
        IR peak indices.

    aligned_red_valleys1_idx : list
        First red valley indices.

    aligned_red_vallleys2_idx : list
        Second red valley indices.

    aligned_ir_valleys1_idx : list
        First IR valley indices.

    aligned_ir_valleys2_idx : list
        Second IR valley indices.

    Returns
    -------
    good_red_beats : list
    good_ir_beats : list
    good_red_valley_beats : list of tuples
    good_ir_valley_beats : list of tuples
    """

    good_red_beats = []
    good_ir_beats = []

    good_red_valley_beats = []
    good_ir_valley_beats = []

    for mask_idx in range(len(mask_indices)):

        if mask_indices[mask_idx] == 1:

            good_red_beats.append(aligned_red_beats_idx[mask_idx])

            good_ir_beats.append(aligned_ir_beats_idx[mask_idx])

            good_red_valley_beats.append((aligned_red_valleys1_idx[mask_idx], aligned_red_vallleys2_idx[mask_idx]))

            good_ir_valley_beats.append((aligned_ir_valleys1_idx[mask_idx], aligned_ir_valleys2_idx[mask_idx]))

    return (good_red_beats, good_ir_beats, good_red_valley_beats, good_ir_valley_beats)



# ----------------------------- AC and DC Calculation -----------------------------------
def calculate_AC(filtered_signal, peaks, valleys):
    """
    Calculate AC values.
    
    Returns
    -------
    ac_values : array
        AC values.
    """
    ac_values = []
    for i in range(0, len(valleys)):
        valley1 = valleys[i][0]
        valley2 = valleys[i][1]
        ac = filtered_signal[peaks[i]] - filtered_signal[valley1]
        ac_values.append(ac)

    ac_values = np.array(ac_values)

    return ac_values

def calculate_DC(raw_signal, valleys):
    """
    Calculate DC values.
    
    Returns
    -------
    dc_values : array
        DC values.
    """
    dc_values = []
    
    for i in range(len(valleys)):
        valley1 = valleys[i][0]
        valley2 = valleys[i][1]
        dc_signal = raw_signal[valley1:valley2]
        dc_mean = np.mean(dc_signal) # baseline component
        dc_values.append(dc_mean)

    dc_values = np.array(dc_values)
    return dc_values

# -------------------------------------------- R calculation ---------------------------------
def calculate_R(ac_red, ac_ir, dc_red, dc_ir):
    """
    Calculate R values.
    
    Returns
    -------
    r_values : array
        R values.
    """
    # R = (AC(red)/DC(red)) / (AC(infrared)/DC(infrared))    
    r_values = []
    for i in range(len(ac_red)):
        r = (ac_red[i]/dc_red[i]) / (ac_ir[i]/dc_ir[i])
        r_values.append(r)
    
    r_values = np.array(r_values)
    return r_values

# -------------------------------------------- SPO2 calculation (linear & quadratic) ---------------------------------
def spo2_calculator(r_values):
    """
    Calculate SPO2 values.
    
    Returns
    -------
    spo2_values_linear : array
        SPO2 values (linear).
    spo2_values_quadratic : array
        SPO2 values (quadratic).
    spo2_avg_linear : float
        Average SPO2 (linear).
    spo2_avg_quadratic : float
        Average SPO2 (quadratic).
    spo2_median_linear : float
        Median SPO2 (linear).
    spo2_median_quadratic : float
        Median SPO2 (quadratic).
    """
    spo2_values_linear = []
    spo2_values_quadratic = []
    for r in r_values:
        spo2_linear = 104 - 17 * r
        spo2_quad = -0.375 * r**2 - 20 * r + 108
        spo2_values_linear.append(spo2_linear)
        spo2_values_quadratic.append(spo2_quad)
    
    spo2_values_linear = np.array(spo2_values_linear)
    spo2_values_quadratic = np.array(spo2_values_quadratic)
    spo2_avg_linear = np.mean(spo2_values_linear)
    spo2_avg_quadratic = np.mean(spo2_values_quadratic)
    spo2_median_linear = np.median(spo2_values_linear)
    spo2_median_quadratic = np.median(spo2_values_quadratic)
    return spo2_values_linear, spo2_values_quadratic, spo2_avg_linear, spo2_avg_quadratic, spo2_median_linear, spo2_median_quadratic