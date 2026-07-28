import numpy as np

def extract_cycle_periods(valleys, fs):

    # Compute individual cycle periods (valley-to-valley)
    cycle_periods = []
    for i in range(len(valleys) - 1):
        period = (valleys[i+1] - valleys[i]) / fs
        cycle_periods.append(period)

    avg = np.mean(cycle_periods)
    
    return cycle_periods, avg