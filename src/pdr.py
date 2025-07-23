import math
import numpy as np
import lowpass as lp
import pandas as pd

def pull_data (dir_name, file_name):
    f = open(dir_name + '/' + file_name + '.csv')
    Hx = []
    Hy = []
    Hz = []
    SVM = []
    timestamps = []
    for line in f:
        value = line.split(',')
        if len(value) == 4:
            timestamps.append(value[0])
            Hx.append(float(value[1]))
            Hy.append(float(value[2]))
            Hz.append(float(value[3]))
            value = math.sqrt(float(value[0])**2 + float(value[1])**2 + float(value[2])**2)
            SVM.append(value)
    f.close()
    return np.array(Hx), np.array(Hy), np.array(Hz), np.array(SVM), np.array(timestamps)

def data_corrected (Hx, Hy, Hz, SVM): #? to center around 0
    Hx = Hx - np.mean(Hx)
    Hy = Hy - np.mean(Hy)
    Hz = Hz - np.mean(Hz)
    SVM = SVM - np.mean(SVM)
    return Hx, Hy, Hz, SVM

acc_data = pull_data('data', 'Accelerometer')
gyro_data = pull_data('data', 'Gyroscope')

acc = np.array ([data_corrected(*acc_data)])

acc_filtered = np.array([lp.lowpass_filter(acc[0][0], 0.1, 50)
                                , lp.lowpass_filter(acc[0][1], 0.1, 50)
                                , lp.lowpass_filter(acc[0][2], 0.1, 50)])


# rotation matrices

def R_x(x):
    return np.array([[1, 0, 0],
                     [0, np.cos(x), -np.sin(x)],
                     [0, np.sin(x), np.cos(x)]])
    

def R_y(y):
    return np.array([[np.cos(y), 0, np.sin(y)],
                     [0, 1, 0],
                     [-np.sin(y), 0, np.cos(y)]])

def R_z(z):
    return np.array([[np.cos(z), -np.sin(z), 0],
                     [np.sin(z), np.cos(z), 0],
                     [0, 0, 1]])
    

