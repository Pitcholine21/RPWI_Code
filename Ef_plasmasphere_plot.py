import numpy as np
import datetime as dt
import spiceypy as spice
import Ef_utilities as Ef
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.labelcolor'] = 'linecolor'
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['font.family'] = 'Serif'

def parse_datetime(s):
    # Handles both "YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DD HH:MM:SS.sss"
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def parse_file(path, n):
    # n is the number of variables in the file

    lp_epochs = []
    lists = [[] for _ in range(n-1)]  # Create a list for each variable

    with open(path, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            lp_epochs.append(parse_datetime(parts[0]))
            for i in range(1, n):
                lists[i-1].append(float(parts[i]))

    lp_epochs = np.array(lp_epochs)
    lists = [np.array(lst) for lst in lists]
    return lp_epochs, lists

def insert_NaNs(epoch, lists):
    new_epoch = []
    new_lists = [[] for _ in lists]  # Create a list for each list to hold NaN values

    for i in range(len(epoch)):
        new_epoch.append(epoch[i])
        for j in range(len(lists)):
            new_lists[j].append(lists[j][i])
        
        if i < len(epoch) - 1:
            dt_sec = (epoch[i+1] - epoch[i]).total_seconds()
            if dt_sec > 1:
                # Insert NaN to break the line
                new_epoch.append(epoch[i] + dt.timedelta(seconds=1))
                for j in range(len(lists)):
                    new_lists[j].append(np.nan)

    return np.array(new_epoch), [np.array(lst) for lst in new_lists]

spice.furnsh("SPICE/JUICE/kernels/mk/juice_ops.tm")
spice.furnsh("SPICE/gsm_frame.tf")

folder_LP = "Olivier_RPWI/Plasmasphere_data_files/LP/"
filename_LP = "Efield_20240820T120000_20240821T121500_downsample200_caractime300.txt"
path_LP = folder_LP + filename_LP

# Extract downsample_factor and carac_time from the LP filename
m = re.search(r'downsample(\d+)_caractime(\d+)', path_LP)
if m:
    downsample_factor = int(m.group(1))
    carac_time = int(m.group(2))

lp_epochs = []
Ex, Ey, Ez = [], [], []
Exreg, Eyreg, Ezreg = [], [], []

lp_epochs, lists = parse_file(path_LP, 7)

lp_epochs, lists = insert_NaNs(lp_epochs, lists)
Ex, Ey, Ez, Exreg, Eyreg, Ezreg = lists



# PLOTTING E-F AFTER REGULAR CALIBRATION
Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lp_epochs, Ex*1e3, label='Ex (sliding cal)', lw=1.1)
ax.plot(lp_epochs, Ey*1e3, label='Ey (sliding cal)', lw=1.1)
ax.plot(lp_epochs, Ez*1e3, label='Ez (sliding cal)', lw=1.1)
ax.plot(lp_epochs, Emag*1e3, label='|E| (sliding cal)', color = 'red', lw=1.1)
ax.set_title(f"Caracteristic time: {carac_time} seconds, downsample factor: {downsample_factor}")
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.04, -0.02)
ticks = ax.get_xticks()
ticks = Ef.convert_1970(ticks)
distances = Ef.get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Electric Field (mV/m)')
ax.legend()
ax.grid()

# PLOTTING E-F AFTER NEW CALIBRATION
Emagreg = np.sqrt(Exreg**2 + Eyreg**2 + Ezreg**2)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lp_epochs, Exreg*1e3, label='Ex (regular cal)', lw=1.1)
ax.plot(lp_epochs, Eyreg*1e3, label='Ey (regular cal)', lw=1.1)
ax.plot(lp_epochs, Ezreg*1e3, label='Ez (regular cal)', lw=1.1)
ax.plot(lp_epochs, Emagreg*1e3, label='|E| (regular cal)', color = 'red', lw=1.1)
ax.set_title(f"Downsample factor: {downsample_factor}")
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.04, -0.02)
ticks = ax.get_xticks()
ticks = Ef.convert_1970(ticks)
distances = Ef.get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Electric Field (mV/m)')
ax.legend()
ax.grid()

spice.kclear()

plt.show()