import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# load the accelerometer data from CSV

# the CSV file currently has the following columns: timestamps, imei, acceleration_x, acceleration_y, acceleration_z

# at the moment there is no gyroscope or magnetometer data, but the following code acts like it does 

df = pd.read_csv('data/Accelerometer.csv')

accel = df[['acceleration_x', 'acceleration_y', 'acceleration_z']].values
gyros = df[['gyroscope_x', 'gyroscope_y', 'gyroscope_z']].values
mags = df[['magnetometer_x', 'magnetometer_y', 'magnetometer_z']].values
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
mags_filtered = butter_lowpass_filter(mags)

# step detection

acc_mag = np.linalg.norm(accel_filtered, axis=1)
peaks, _ = find_peaks(acc_mag, height=0.5, distance=20)  # adjust height and distance as needed
step_timestamps = timestamps[peaks]

step_length = 0.7 # average step length in meters 

### TODO: implement formulae to compute step length based on accelerometer data or height/weight of the user

# heading estimation using gyroscope and magnetometer data
# gyroscope integration for short-term, magnetometer for long-term stability

gyro_z = gyros_filtered[:, 2]
dt = 1.0 / 50  # assuming 50 Hz sampling rate

mag_x = mags_filtered[:, 0]
mag_y = mags_filtered[:, 1]
mag_heading = np.arctan2(mag_y, mag_x)

alpha = 0.98  # gyroscope weight (tune as needed)
heading = np.zeros(len(gyro_z))
heading[0] = mag_heading[0]

for i in range(1, len(gyro_z)):
    gyro_delta = gyro_z[i] * dt
    heading_gyro = heading[i-1] + gyro_delta
    mag_h = mag_heading[i]
    heading[i] = alpha * heading_gyro + (1 - alpha) * mag_h
    if heading[i] > np.pi:
        heading[i] -= 2 * np.pi
    elif heading[i] < -np.pi:
        heading[i] += 2 * np.pi

step_headings = heading[peaks]

# reconstructing the path (the actual PDR algorithm)


x = [0.0]
y = [0.0]
for i in range(1, len(peaks)):
    dx = step_length * np.cos(step_headings[i])
    dy = step_length * np.sin(step_headings[i])
    x.append(x[-1] + dx)
    y.append(y[-1] + dy)
    
x = np.array(x)
y = np.array(y)

# kalman filter (prediction only, no GPS correction **at the moment**)

state = np.array([0.0, 0.0, step_headings[0]])
P = np.eye(3) * 0.1  # initial state uncertainty
Q = np.eye(3) * 0.01  # process noise covariance
R = np.eye(3) * 0.1  # measurement noise covariance

kalman_states = [state.copy()]

for i in range(1, len(x)):
    d_heading = step_headings[i] - step_headings[i-1]
    state_pred = state + np.array([
        step_length * np.sin(step_headings[i]),
        step_length * np.cos(step_headings[i]),
        d_heading
    ])
    P = P + Q
    # no external correction since no GPS
    state = state_pred
    kalman_states.append(state.copy())

kalman_states = np.array(kalman_states)
x_kalman = kalman_states[:,0]
y_kalman = kalman_states[:,1]

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='PDR Path', marker='o')
plt.plot(x_kalman, y_kalman, label='PDR + Kalman (prediction only)', linestyle='--', marker='x')
plt.title('Estimated Pedestrian Trajectory with Gyroscope & Magnetometer')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.legend()
plt.grid()
plt.show()