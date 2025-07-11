import numpy as np
import datetime as dt
import spiceypy as spice
import Ef_utilities as Ef
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utilities import print_entire
import re

spice.furnsh("SPICE/JUICE/kernels/mk/juice_ops.tm")
spice.furnsh("SPICE/gsm_frame.tf")

output_filename = "Efield_20240820T120000_20240821T121500_downsample200_caractime300_Exponential.txt"

lp_epochs = []
Ex, Ey, Ez, Emag = [], [], [], []
Exreg, Eyreg, Ezreg, Emagreg = [], [], [], []

def parse_datetime(s):
    # Handles both "YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DD HH:MM:SS.sss"
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

with open(output_filename, "r") as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue
        lp_epochs.append(parse_datetime(parts[0]))
        Ex.append(float(parts[1]))
        Ey.append(float(parts[2]))
        Ez.append(float(parts[3]))
        Emag.append(float(parts[4]))
        Exreg.append(float(parts[5]))
        Eyreg.append(float(parts[6]))
        Ezreg.append(float(parts[7]))
        Emagreg.append(float(parts[8]))

lp_epochs = np.array(lp_epochs)
Ex = np.array(Ex)
Ey = np.array(Ey)
Ez = np.array(Ez)
Emag = np.array(Emag)
Exreg = np.array(Exreg)
Eyreg = np.array(Eyreg)
Ezreg = np.array(Ezreg)
Emagreg = np.array(Emagreg)

# Extract downsample_factor and carac_time from the filename
m = re.search(r'downsample(\d+)_caractime(\d+)', output_filename)
if m:
    downsample_factor = int(m.group(1))
    carac_time = int(m.group(2))

# Insert NaN between points where time gap > 1 second
new_lp_epochs = []
new_Ex, new_Ey, new_Ez, new_Emag = [], [], [], []
new_Exreg, new_Eyreg, new_Ezreg, new_Emagreg = [], [], [], []

for i in range(len(lp_epochs)):
    new_lp_epochs.append(lp_epochs[i])
    new_Ex.append(Ex[i])
    new_Ey.append(Ey[i])
    new_Ez.append(Ez[i])
    new_Emag.append(Emag[i])
    new_Exreg.append(Exreg[i])
    new_Eyreg.append(Eyreg[i])
    new_Ezreg.append(Ezreg[i])
    new_Emagreg.append(Emagreg[i])
    if i < len(lp_epochs) - 1:
        dt_sec = (lp_epochs[i+1] - lp_epochs[i]).total_seconds()
        if dt_sec > 1:
            # Insert NaN to break the line
            new_lp_epochs.append(lp_epochs[i] + dt.timedelta(seconds=1))
            new_Ex.append(np.nan)
            new_Ey.append(np.nan)
            new_Ez.append(np.nan)
            new_Emag.append(np.nan)
            new_Exreg.append(np.nan)
            new_Eyreg.append(np.nan)
            new_Ezreg.append(np.nan)
            new_Emagreg.append(np.nan)

lp_epochs = np.array(new_lp_epochs)
Ex = np.array(new_Ex)
Ey = np.array(new_Ey)
Ez = np.array(new_Ez)
Emag = np.array(new_Emag)
Exreg = np.array(new_Exreg)
Eyreg = np.array(new_Eyreg)
Ezreg = np.array(new_Ezreg)
Emagreg = np.array(new_Emagreg)

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