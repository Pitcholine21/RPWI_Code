import struct
import cdflib
import numpy as np
import datetime as dt
import spiceypy as spice
from scipy.signal import butter, filtfilt

R_E = 6371.2  # Earth radius in km

def filter_out_start_config_noise(data, epoch, removal_width = 32, new_config_delay = 0.1):
    # Create a mask for the data
    noise_mask = np.zeros(len(data))

    # Mask the next samples when dt is above new_config_delay, for which a new config. has been made
    delta_t = np.diff(epoch)
    delta_t = [dt.total_seconds() for dt in delta_t]

    i = 0
    for dt in delta_t:
        if dt > new_config_delay:
            noise_mask[i:i+removal_width] = 1
        i +=1

    noise_mask[:removal_width] = 1

    data[noise_mask == 1] = np.nan

    return data

def multiply_lists_by_33matrix(list1, list2, list3, matrix):
    res1 = np.zeros(len(list1))
    res2 = np.zeros(len(list1))
    res3 = np.zeros(len(list1))
    for i in range(len(list1)):
        vector = np.array([list1[i], list2[i], list3[i]])
        res1[i] = vector[0]*matrix[0, 0] + vector[1]*matrix[0, 1] + vector[2]*matrix[0, 2]
        res2[i] = vector[0]*matrix[1, 0] + vector[1]*matrix[1, 1] + vector[2]*matrix[1, 2]
        res3[i] = vector[0]*matrix[2, 0] + vector[1]*matrix[2, 1] + vector[2]*matrix[2, 2]
    return res1, res2, res3

def multiply_lists_by_44matrix(list1, list2, list3, list4, matrix):
    res1 = np.zeros(len(list1))
    res2 = np.zeros(len(list1))
    res3 = np.zeros(len(list1))
    res4 = np.zeros(len(list1))
    for i in range(len(list1)):
        vector = np.array([list1[i], list2[i], list3[i], list4[i]])
        res1[i] = vector[0]*matrix[0, 0] + vector[1]*matrix[0, 1] + vector[2]*matrix[0, 2] + vector[3]*matrix[0, 3]
        res2[i] = vector[0]*matrix[1, 0] + vector[1]*matrix[1, 1] + vector[2]*matrix[1, 2] + vector[3]*matrix[1, 3]
        res3[i] = vector[0]*matrix[2, 0] + vector[1]*matrix[2, 1] + vector[2]*matrix[2, 2] + vector[3]*matrix[2, 3]
        res4[i] = vector[0]*matrix[3, 0] + vector[1]*matrix[3, 1] + vector[2]*matrix[3, 2] + vector[3]*matrix[3, 3]
    return res1, res2, res3, res4

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    valid_indices = ~np.isnan(data)
    y = filtfilt(b, a, data[valid_indices])
    return y

def convert_binstr_to_double_precision(bin_str_array):
    doubles = []
    for bin_str in bin_str_array:
        # Convert binary string to 8-byte integer
        try :
            int_val = int(bin_str, 2)
            bytes_val = int_val.to_bytes(8, byteorder='big')
            double_val = struct.unpack('>d', bytes_val)[0]
            doubles.append(double_val)
        except ValueError as e:
            doubles.append(np.nan)
    return np.array(doubles)

def extract_double_from_columns(data, col_start, col_end):
    # Slice relevant columns: shape (len_JMAG, 8)
    bytes_slice = data[:, col_start:col_end]
    # Transpose and flatten to match MATLAB logic
    bin_strs = [''.join(f'{byte:08b}' for byte in row) for row in bytes_slice]
    return convert_binstr_to_double_precision(bin_strs)

# Magnetopause model (Shue et al. 1997)
def magnetopause(P_sw, theta):
    R0 = 10.22 * (P_sw / 1.0)**(-1/6)  # subsolar point in Earth radii
    alpha = 0.5  # typical value
    r = R0 * (2 / (1 + np.cos(theta)))**alpha
    return r

# Bow Shock model (Farris & Russell 1994)
def bow_shock(P_sw, theta):
    R0 = 14.5 * (P_sw / 1.0)**(-1/6)  # subsolar point in Earth radii
    epsilon = 0.8  # shape parameter
    r = (R0 * (1 + epsilon)) / (1 + epsilon * np.cos(theta))
    return r

def find_factor(list1, list2):
    # Find the factor with multiplies list1 so that both lists have the same mean absolute deviation from their mean
    mean1 = np.nanmean(np.abs(list1 - np.nanmean(list1)))
    mean2 = np.nanmean(np.abs(list2 - np.nanmean(list2)))
    factor = mean2 / mean1
    return factor

def convert_1970(epochs):
    datetimes = []
    for epoch in epochs:
        datetimes.append(dt.datetime(1970, 1, 1) + dt.timedelta(days=epoch))
    return datetimes

def get_JUICE_distance(epochs):
    dist_list = []
    for epoch in epochs:
        # Get the distance from JUICE to Earth in Earth radii
        et = spice.datetime2et(epoch)
        pos, _ = spice.spkpos('JUICE', et, 'GSM', 'NONE', 'EARTH')
        pos = np.array(pos)
        pos_norm = np.linalg.norm(pos)
        pos_norm_earth_radii = pos_norm / R_E
        dist_list.append(float(pos_norm_earth_radii))
    return dist_list

def print_info(cdf):

    for key, value in cdf.globalattsget().items():
        print(f"Global attribute: {key} = {value}")

    for variable in cdf.cdf_info().zVariables:
        print()
        print(f"Variable: {variable}")
        print(f"Shape: {cdf.varget(variable).shape}")
        print(f"Data type: {cdf.varinq(variable).Data_Type_Description}")
        for key, value in cdf.varattsget(variable).items():
            print(f"Variable attribute: {key} = {value}")

def print_entire(array):
    for i in range(len(array)):
        print(f"Index {i}: {array[i]}")
    return

# Rotation matrix to go from JMAG OBS frame to JUICE frame
R_JMAG = np.array([
    [-7.77145961*1e-1,  8.39299198*1e-17,   -6.29320391*1e-1],
    [-9.51729314*1e-17, -1.00000000*1e0,    -1.58371803*1e-17],
    [-6.29320391*1e-1,  4.75864657*1e-17,   7.77145961*1e-1]])

diff2volt = np.array([
    [-1.0, -1.0, -1.0, 1.0],
    [0.0, -1.0, -1.0, 1.0],
    [0.0, 0.0, -1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0]])

volt2diff = np.array([
    [-1.0, 1.0, 0.0, 0.0],
    [0.0, -1.0, 1.0, 0.0],
    [0.0, 0.0, -1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0]])

volt2E = np.array([
    [0.1852, 0.1923, 0.1917],
    [0.1320, -0.0112, -0.0322],
    [0.0, 0.1398, 0.0]])

E2volt = np.linalg.inv(volt2E)

TM2diff = np.array([5.15*1e-6, 4.97*1e-6, 5.07*1e-6, 9.94*1e-5]) # Coefficients to go from telemetry units to differentials

probe_dist = 1e-3*np.array([6287.3, 7152.0, 9611.4]) # in m, LP1-LP2, LP2-LP3, LP3-LP4