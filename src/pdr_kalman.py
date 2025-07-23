import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


# load the accelerometer data from CSV

# the CSV file currently has the following columns: timestamps, imei, acceleration_x, acceleration_y, acceleration_z

# at the moment there is no gyroscope data, but the following code acts like it does 

df = pd.read_csv('data/Accelerometer.csv')

accel = df[['acceleration_x', 'acceleration_y', 'acceleration_z']].values
gyros = df[['gyroscope_x', 'gyroscope_y', 'gyroscope_z']].values
timestamps = df['timestamps'].values

# butterworth lowpass filter to smoothen the accelerometer/gyroscope data
# attenuates high frequencies (i.e. noise)

# cutoff = the frequency at which the filter starts attenuating the signal
# fs = sampling frequency (i.e. how often the data is sampled)
# order = controls how sharp the transition is, a higher order means a sharper 
#         transition, but may introduce delay

# !!! NOT IDEAL FOR PRESERVING SHARP EDGES OR FAST MOVEMENTS (e.g. punches, impacts) !!!    
# we're very interested in those, but for now we just want to detect steps :)

def butter_lowpass_filter(data, cutoff=3, fs=50, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data, axis=0)
    return filtered

accel_filtered = butter_lowpass_filter(accel)
gyros_filtered = butter_lowpass_filter(gyros)

# step detection

from scipy.signal import find_peaks

acc_mag = np.linalg.norm(accel_filtered, axis=1)

peaks, _ = find_peaks(acc_mag, height=0.5, distance=50)

step_timestamps = timestamps[peaks]


