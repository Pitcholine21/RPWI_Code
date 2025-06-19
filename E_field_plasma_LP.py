print("Running E-field plasma LP correction script")

import cdflib
import numpy as np
import datetime as dt
import spiceypy as spice
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Ef_utilities import *
from utilities import print_entire

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.labelcolor'] = 'linecolor'
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['font.family'] = 'Serif'
plt.ion()

def exponential(t, carac_time):
    return np.exp(-(t / carac_time)**2)

def box(t, carac_time):
    return 1

def correct_for_spacecraft_charge(U123, U4, epochs, carac_time, weight_function=exponential):
    # carac_time is the characteristic time of the exponential in seconds
    # Calculate the average weighed by the exponential function given

    nb_epochs = len(epochs)

    U123_corrected = np.zeros(nb_epochs)

    start_time = dt.datetime.now()

    remainings = []
    timestamps = []
    durations = []

    for i in range(nb_epochs):
        curr_time = dt.datetime.now()
        epoch = epochs[i]
        mean_numerator = 0
        mean_denominator = 0
        timestamps.append(curr_time)
        seconds = (curr_time - start_time).total_seconds()
        remaining = seconds*(nb_epochs-i-1)/(i+1)
        remainings.append(remaining)
        print(f"Correcting for time {epoch} ({i+1}/{nb_epochs}), estimated time remaining at {remaining:.2f} seconds")
        for j in range(i,nb_epochs):
            time_delta = (epochs[j]-epoch).total_seconds()
            if time_delta >= 5*carac_time:
                #print(f"Breaking at {epochs[j]} in incremental loop")
                break
            if np.isfinite(U4[j]) and np.isfinite(U123[j]):
                weight = weight_function(time_delta, carac_time)
                mean_numerator += (U4[j] - U123[j]) * weight   
                mean_denominator += weight
        for j in range(i-1, -1, -1):
            time_delta = (epochs[j]-epoch).total_seconds()
            if time_delta <= -5*carac_time:
                #print(f"Breaking at {epochs[j]} in decremental loop")
                break
            if np.isfinite(U4[j]) and np.isfinite(U123[j]):
                weight = weight_function(time_delta, carac_time)
                mean_numerator += (U4[j] - U123[j]) * weight   
                mean_denominator += weight
        if mean_denominator != 0:
            U123_corrected[i] = U123[i] + mean_numerator / mean_denominator
        else:
            U123_corrected[i] = U123[i]
        durations.append((dt.datetime.now() - curr_time).total_seconds())

    """
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, remainings, label='Estimated time remaining')
    plt.xlabel('Timestamp')
    plt.ylabel('Estimated time remaining (seconds)')
    plt.title('Estimated Time Remaining for Correction')
    plt.legend()
    plt.grid()

    num_bins = 10

    timestamps_float = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])
    bin_edges = np.linspace(timestamps_float.min(), timestamps_float.max(), num_bins + 1)
    bin_indices = np.digitize(timestamps_float, bin_edges) - 1  # bins: 0 to num_bins-1

    colors = plt.cm.viridis(np.linspace(0, 1, num_bins))

    # Create a stacked bar plot (histogram) with multicolored "bins
    plt.figure(figsize=(10, 5))
    num_bins_hist = 30
    # For each time bin, get the durations that fall into it
    stacked_counts = []
    bins = np.linspace(np.min(durations), np.max(durations), num_bins_hist + 1)
    for b in range(num_bins):
        mask = (bin_indices == b)
        counts, _ = np.histogram(np.array(durations)[mask], bins=bins)
        stacked_counts.append(counts)
    stacked_counts = np.array(stacked_counts)

    # Plot stacked bars
    bottom = np.zeros(num_bins_hist)
    for b in range(num_bins):
        plt.bar(
            0.5 * (bins[:-1] + bins[1:]),
            stacked_counts[b],
            width=(bins[1] - bins[0]),
            bottom=bottom,
            color=colors[b],
            edgecolor='black',
            label=f'Time bin {b+1}'
        )
        bottom += stacked_counts[b]

    plt.xlabel('Duration per iteration (seconds)')
    plt.ylabel('Count')
    plt.title('Stacked Histogram of Correction Durations per Iteration (by time bin)')
    plt.legend()
    plt.grid()
    """

    return U123_corrected

R_E = 6371  # Earth radius

carac_time = 300
downsample_factor = 300

start_time = dt.datetime(2024, 8, 20, 21, 0, 0)
end_time = dt.datetime(2024, 8, 20, 21, 15, 0)

start = dt.datetime.now()

# Get cdfs and JUICE state vectors

cdfs = []

print("Loading CDF files...")

cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T180737_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T184752_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T192916_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T200931_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T205007_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T212819_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T220245_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T223650_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T231519_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T235531_V03.cdf'))

print(f"Got {len(cdfs)} CDF files")

# Get E-field data

U12 = np.array([])
U23 = np.array([])
U34 = np.array([])
U40 = np.array([])
lp_epochs = np.array([])

for idx, cdf in enumerate(cdfs):
    lp_epoch = cdf.varget('Epoch')
    lp_epoch = cdflib.cdfepoch.to_datetime(lp_epoch)
    lp_epoch = np.array(lp_epoch, dtype='datetime64[ms]').astype('O')
    lp_epochs = np.concatenate((lp_epochs, lp_epoch))

    u12 = cdf.varget('LP_DATA')[:,0]
    u23 = cdf.varget('LP_DATA')[:,1]
    u34 = cdf.varget('LP_DATA')[:,2]
    u40 = cdf.varget('LP_DATA')[:,3]

    U12 = np.concatenate((U12, u12))
    U23 = np.concatenate((U23, u23))
    U34 = np.concatenate((U34, u34))
    U40 = np.concatenate((U40, u40))

print("Got LP data")

U12 = U12 * TM2diff[0]
U23 = U23 * TM2diff[1]
U34 = U34 * TM2diff[2]
U40 = U40 * TM2diff[3]

U12 = filter_out_start_config_noise(U12, lp_epochs)
U23 = filter_out_start_config_noise(U23, lp_epochs)
U34 = filter_out_start_config_noise(U34, lp_epochs)
U40 = filter_out_start_config_noise(U40, lp_epochs)

print("Got filtered LP data")

# Get the indices for the epochs within the specified time range
start_idx = np.searchsorted(lp_epochs, start_time)
end_idx = np.searchsorted(lp_epochs, end_time)

U12 = U12[start_idx:end_idx:downsample_factor]
U23 = U23[start_idx:end_idx:downsample_factor]
U34 = U34[start_idx:end_idx:downsample_factor]
U40 = U40[start_idx:end_idx:downsample_factor]
lp_epochs = lp_epochs[start_idx:end_idx:downsample_factor]

print("Downsampled and truncated LP data")

"""
distances = []

spice.furnsh("SPICE/JUICE/kernels/mk/juice_ops.tm")
spice.furnsh("SPICE/gsm_frame.tf")

for epoch in lp_epochs:
    # Get the distance from JUICE to Earth in earth radii
    et = spice.str2et(epoch.isoformat())
    pos, _ = spice.spkpos('JUICE', et, 'GSM', 'NONE', 'EARTH')
    pos = np.array(pos)
    pos_norm = np.linalg.norm(pos)
    pos_norm_earth_radii = pos_norm / R_E
    distances.append(pos_norm_earth_radii)

spice.kclear()

print("Got distances from JUICE to Earth")
"""

plt.figure(figsize=(12, 6))
plt.plot(lp_epochs, U12, label='U12', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U23, label='U23', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U34, label='U34', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U40, label='U40', marker='o', linestyle='', markersize=1)
plt.title(f"Caracteristic time: {carac_time} seconds, downsample factor: {downsample_factor}")
plt.xlabel('Epoch')
plt.ylabel('Voltage difference (V)')
plt.legend()
plt.grid()

U1, U2, U3, U4 = multiply_lists_by_44matrix(U12, U23, U34, U40, diff2volt)

"""
# Plot U1 and U4 over the entire time period
center_time = dt.datetime(2024, 8, 20, 22, 0, 0)
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axs[0].plot(lp_epochs, U1, label='U1')
axs[0].plot(lp_epochs, U4, label='U4')
axs[0].set_title('U1 and U4 over time')
axs[0].set_ylabel('Voltage (V)')
axs[0].legend()
axs[0].grid()

# Prepare exponential weights centered at center_time
exp_weights = exponential(np.array([(dt.datetime.combine(lp_epochs[i].date(), lp_epochs[i].time()) - center_time).total_seconds() for i in range(len(lp_epochs))]), carac_time)

# Plot U1*exp and U4*exp
axs[1].plot(lp_epochs, U1 * exp_weights, label='U1 * exponential(t)')
axs[1].plot(lp_epochs, U4 * exp_weights, label='U4 * exponential(t)')
axs[1].plot(lp_epochs, exp_weights, label='Exponential(t)', linestyle='--', color='red')
axs[1].set_title('U1 and U4 weighted by exponential(t)')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Weighted Voltage (V)')
axs[1].legend()
axs[1].grid()

plt.tight_layout()

# Calculate and print average of (U4 - U1) weighted by exponential(t)
weighted_diff = (U4 - U1) * exp_weights
avg_weighted_diff = np.nansum(weighted_diff) / np.nansum(exp_weights)
axs[1].text(0.01, 0.90, f'Weighted average of U4-U1: {avg_weighted_diff:.4f} V', transform=axs[1].transAxes, fontsize=12, verticalalignment='top')

print(f"Weighted average of U4-U1: {avg_weighted_diff:.4f} V")

plt.show()
"""

print("Got single probe potentials")

plt.figure(figsize=(12, 6))
plt.plot(lp_epochs, U1, label='U1 (uncalibrated)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U2, label='U2 (uncalibrated)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U3, label='U3 (uncalibrated)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U4, label='U4 (uncalibrated)', marker='o', linestyle='', markersize=1)
plt.title(f"Caracteristic time: {carac_time} seconds, downsample factor: {downsample_factor}")
plt.xlabel('Epoch')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid()

mean_diff_U1 = np.nanmean(np.array(U4) - np.array(U1))
mean_diff_U2 = np.nanmean(np.array(U4) - np.array(U2))
mean_diff_U3 = np.nanmean(np.array(U4) - np.array(U3))

U1reg = np.array([float(u) + float(mean_diff_U1) for u in U1])
U2reg = np.array([float(u) + float(mean_diff_U2) for u in U2])
U3reg = np.array([float(u) + float(mean_diff_U3) for u in U3])

weight_function_used = "Exponential"
U1 = correct_for_spacecraft_charge(U1, U4, lp_epochs, carac_time, weight_function=exponential)
U2 = correct_for_spacecraft_charge(U2, U4, lp_epochs, carac_time, weight_function=exponential)
U3 = correct_for_spacecraft_charge(U3, U4, lp_epochs, carac_time, weight_function=exponential)

print("Corrected for spacecraft charge")

plt.figure(figsize=(12, 6))
plt.plot(lp_epochs, U1, label='U1 (sliding cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U2, label='U2 (sliding cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U3, label='U3 (sliding cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U4, label='U4 (sliding cal)', marker='o', linestyle='', markersize=1)
plt.title(f"Caracteristic time: {carac_time} seconds, downsample factor: {downsample_factor}, weight function: {weight_function_used}")
plt.xlabel('Epoch')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid()

plt.figure(figsize=(12, 6))
plt.plot(lp_epochs, U1reg, label='U1 (regular cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U2reg, label='U2 (regular cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U3reg, label='U3 (regular cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U4, label='U4 (regular cal)', marker='o', linestyle='', markersize=1)
plt.title(f"Downsample factor: {downsample_factor}")
plt.xlabel('Epoch')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid()

U12, U23, U34, U40 = multiply_lists_by_44matrix(U1, U2, U3, U4, volt2diff)
U12reg, U23reg, U34reg, U40reg = multiply_lists_by_44matrix(U1reg, U2reg, U3reg, U4, volt2diff)

print("Went back to differentials")

plt.figure(figsize=(12, 6))
plt.plot(lp_epochs, U12, label='U12 (sliding cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U23, label='U23 (sliding cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U34, label='U34 (sliding cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U40, label='U40 (sliding cal)', marker='o', linestyle='', markersize=1)
plt.title(f"Caracteristic time: {carac_time} seconds, downsample factor: {downsample_factor}, weight function: {weight_function_used}")
plt.xlabel('Epoch')
plt.ylabel('Voltage difference (V)')
plt.legend()
plt.grid()

plt.figure(figsize=(12, 6))
plt.plot(lp_epochs, U12reg, label='U12 (regular cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U23reg, label='U23 (regular cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U34reg, label='U34 (regular cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, U40reg, label='U40 (regular cal)', marker='o', linestyle='', markersize=1)
plt.title(f"Downsample factor: {downsample_factor}")
plt.xlabel('Epoch')
plt.ylabel('Voltage difference (V)')
plt.legend()
plt.grid()

Ex, Ey, Ez = multiply_lists_by_33matrix(U12, U23, U34, volt2E)
Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
Exreg, Eyreg, Ezreg = multiply_lists_by_33matrix(U12reg, U23reg, U34reg, volt2E)
Emagreg = np.sqrt(Exreg**2 + Eyreg**2 + Ezreg**2)

print("Got E field")

plt.figure(figsize=(12, 6))
plt.plot(lp_epochs, Ex*1e3, label='Ex (sliding cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, Ey*1e3, label='Ey (sliding cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, Ez*1e3, label='Ez (sliding cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, Emag*1e3, label='|E| (sliding cal)', linestyle='--', color = 'red')
plt.title(f"Caracteristic time: {carac_time} seconds, downsample factor: {downsample_factor}, weight function: {weight_function_used}")
plt.xlabel('Epoch')
plt.ylabel('Electric Field (mV/m)')
plt.legend()
plt.grid()

plt.figure(figsize=(12, 6))
plt.plot(lp_epochs, Exreg*1e3, label='Ex (regular cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, Eyreg*1e3, label='Ey (regular cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, Ezreg*1e3, label='Ez (regular cal)', marker='o', linestyle='', markersize=1)
plt.plot(lp_epochs, Emagreg*1e3, label='|E| (regular cal)', linestyle='--', color = 'red')
plt.title(f"Downsample factor: {downsample_factor}")
plt.xlabel('Epoch')
plt.ylabel('Electric Field (mV/m)')
plt.legend()
plt.grid()

print("Finished")
print(f"Running time : {(dt.datetime.now()-start).total_seconds()} seconds")
plt.show(block = False)  # Show the plot without blocking the script
input("Press Enter to exit...")  # Keep the plot window open until user closes it
plt.close('all')  # Close all plot windows after user input