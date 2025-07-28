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

def GSM2JUICE(epoch, X_gsm, Y_gsm, Z_gsm):
    X_juice, Y_juice, Z_juice = [], [], []
    for i, ep in enumerate(epoch):
        et = spice.str2et(ep.isoformat())
        matrix = spice.pxform('GSM', 'JUICE_SPACECRAFT', et)
        x_juice, y_juice, z_juice = np.dot(matrix, [X_gsm[i], Y_gsm[i], Z_gsm[i]])
        X_juice.append(x_juice)
        Y_juice.append(y_juice)
        Z_juice.append(z_juice)
    return np.array(X_juice), np.array(Y_juice), np.array(Z_juice)

def JUICE2GSM(epoch, X_juice, Y_juice, Z_juice):
    X_gsm, Y_gsm, Z_gsm = [], [], []
    for i, ep in enumerate(epoch):
        et = spice.str2et(ep.isoformat())
        matrix = spice.pxform('JUICE_SPACECRAFT', 'GSM', et)
        x_gsm, y_gsm, z_gsm = np.dot(matrix, [X_juice[i], Y_juice[i], Z_juice[i]])
        X_gsm.append(x_gsm)
        Y_gsm.append(y_gsm)
        Z_gsm.append(z_gsm)
    return np.array(X_gsm), np.array(Y_gsm), np.array(Z_gsm)

spice.furnsh("SPICE/JUICE/kernels/mk/juice_ops.tm")
spice.furnsh("SPICE/gsm_frame.tf")

FRAME = "GSM"

folder_LP = "Olivier_RPWI/Plasmasphere_data_files/LP/"
filename_LP = "Efield_20240820T120000_20240821T120000_downsample100_caractime300.txt"
path_LP = folder_LP + filename_LP

folder_vxB = "Olivier_RPWI/Plasmasphere_data_files/vxB/"
filename_vxB = "vxB_downsample100.txt"
path_vxB = folder_vxB + filename_vxB

# Extract downsample_factor and carac_time from the LP filename
m = re.search(r'downsample(\d+)_caractime(\d+)', path_LP)
if m:
    downsample_factor = int(m.group(1))
    carac_time = int(m.group(2))

lp_epochs, lists = parse_file(path_LP, 29)

lp_epochs, lists = insert_NaNs(lp_epochs, lists)
U40, U12uncal, U23uncal, U34uncal, U1uncal, U2uncal, U3uncal, corrections1, corrections2, corrections3, U1regcal, U2regcal, U3regcal, U1cal, U2cal, U3cal, U12regcal, U23regcal, U34regcal, U12cal, U23cal, U34cal, Ex, Ey, Ez, Exreg, Eyreg, Ezreg = lists

if FRAME == "GSM":
    print("Changing E-f frame")
    Ex, Ey, Ez = JUICE2GSM(lp_epochs, Ex, Ey, Ez)
    Exreg, Eyreg, Ezreg = JUICE2GSM(lp_epochs, Exreg, Eyreg, Ezreg)

vxB_epochs, lists = parse_file(path_vxB, 12)
Bx, By, Bz, vxSC, vySC, vzSC, vxCorot, vyCorot, EvxBx, EvxBy, EvxBz = lists
vzCorot = np.zeros(len(vxB_epochs))

if FRAME == "JUICE":
    print("Changing vxB frame")
    Bx, By, Bz = GSM2JUICE(vxB_epochs, Bx, By, Bz)
    vxSC, vySC, vzSC = GSM2JUICE(vxB_epochs, vxSC, vySC, vzSC)
    vxCorot, vyCorot, vzCorot = GSM2JUICE(vxB_epochs, vxCorot, vyCorot, vzCorot)
    EvxBx, EvxBy, EvxBz = GSM2JUICE(vxB_epochs, EvxBx, EvxBy, EvxBz)

print("Plotting")
# PLOTTING DIFFERENTIALS BEFORE CALIBRATION
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lp_epochs, U12uncal, label='U12 (uncalibrated)')
ax.plot(lp_epochs, U23uncal, label='U23 (uncalibrated)')
ax.plot(lp_epochs, U34uncal, label='U34 (uncalibrated)')
ax.plot(lp_epochs, U40, label='U40 (uncalibrated)')
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
ax.set_ylabel('Voltage difference (V)')
ax.legend()
ax.grid()

# PLOTTING SINGLE PROBE POTENTIALS BEFORE CALIBRATION
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lp_epochs, U1uncal, label='U1 (uncalibrated)')
ax.plot(lp_epochs, U2uncal, label='U2 (uncalibrated)')
ax.plot(lp_epochs, U3uncal, label='U3 (uncalibrated)')
ax.plot(lp_epochs, U40, label='U4 (uncalibrated)')
ax.set_title(f"Caracteristic time: {carac_time} s, downsample factor: {downsample_factor}")
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.04, -0.02)
ticks = ax.get_xticks()
ticks = Ef.convert_1970(ticks)
distances = Ef.get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Voltage (V)')
ax.legend()
ax.grid()

# PLOTTING SINGLE PROBE POTENTIALS AFTER REGULAR CALIBRATION
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lp_epochs, U1regcal, label='U1 (regular cal)')
ax.plot(lp_epochs, U2regcal, label='U2 (regular cal)')
ax.plot(lp_epochs, U3regcal, label='U3 (regular cal)')
ax.plot(lp_epochs, U40, label='U4 (regular cal)')
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
ax.set_ylabel('Voltage (V)')
ax.legend()
ax.grid()

# PLOTTING SINGLE PROBE POTENTIALS AFTER NEW CALIBRATION
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(lp_epochs, U1cal, label='U1 (sliding cal)')
ax1.plot(lp_epochs, U2cal, label='U2 (sliding cal)')
ax1.plot(lp_epochs, U3cal, label='U3 (sliding cal)')
ax1.plot(lp_epochs, U40, label='U4 (sliding cal)')
ax1.set_title(f"Caracteristic time: {carac_time} s, downsample factor: {downsample_factor}")
ax1.set_xlabel('Epoch \n Distance (R$_E$)')
ax1.xaxis.set_label_coords(-0.04, -0.02)
ticks = ax1.get_xticks()
ticks = Ef.convert_1970(ticks)
distances = Ef.get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax1.annotate(f"{dist:.1f}", xy=(tick, ax1.get_ylim()[0]), xycoords=('data', 'data'),
                 xytext=(0, -20), textcoords='offset points',
                 ha='center', va='top', fontsize=10, rotation=0)
ax1.set_ylabel('Voltage (V)')
ax1.legend(loc='upper left')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(lp_epochs, corrections1, label='U1 correction', color='tab:blue', alpha=0.5)
ax2.plot(lp_epochs, corrections2, label='U2 correction', color='tab:orange', alpha=0.5)
ax2.plot(lp_epochs, corrections3, label='U3 correction', color='tab:green', alpha=0.5)
ax2.set_ylabel('Correction (V)')
ax2.legend(loc='upper right')

# PLOTTING DIFFERENTIALS AFTER REGULAR CALIBRATION
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lp_epochs, U12regcal, label='U12 (regular cal)')
ax.plot(lp_epochs, U23regcal, label='U23 (regular cal)')
ax.plot(lp_epochs, U34regcal, label='U34 (regular cal)')
ax.plot(lp_epochs, U40, label='U40 (regular cal)')
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
ax.set_ylabel('Voltage difference (V)')
ax.legend()
ax.grid()

# PLOTTING DIFFERENTIALS AFTER NEW CALIBRATION
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lp_epochs, U12cal, label='U12 (sliding cal)')
ax.plot(lp_epochs, U23cal, label='U23 (sliding cal)')
ax.plot(lp_epochs, U34cal, label='U34 (sliding cal)')
ax.plot(lp_epochs, U40, label='U40 (sliding cal)')
ax.set_title(f"Caracteristic time: {carac_time} s, downsample factor: {downsample_factor}")
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.04, -0.02)
ticks = ax.get_xticks()
ticks = Ef.convert_1970(ticks)
distances = Ef.get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Voltage difference (V)')
ax.legend()
ax.grid()

# PLOTTING E-F AFTER REGULAR CALIBRATION
Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lp_epochs, Ex*1e3, label='E$_x$ (sliding cal)', lw=1.1)
ax.plot(lp_epochs, Ey*1e3, label='E$_y$ (sliding cal)', lw=1.1)
ax.plot(lp_epochs, Ez*1e3, label='E$_z$ (sliding cal)', lw=1.1)
ax.plot(lp_epochs, Emag*1e3, label='|E| (sliding cal)', color = 'purple', lw=1.1)
ax.set_title(f"Frame: {FRAME}, caracteristic time: {carac_time} s, downsample factor: {downsample_factor}")
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.04, -0.02)
ax2 = ax.twinx()
ax2.plot(lp_epochs, U40, label='U40', color='tab:red', linewidth=1.2)
ax2.set_ylabel('U40 (V)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc = 'upper right')
ticks = ax.get_xticks()
ticks = Ef.convert_1970(ticks)
distances = Ef.get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Electric Field (mV/m)')
ax.legend(loc = 'upper left')
ax.grid()

# PLOTTING E-F AFTER NEW CALIBRATION
Emagreg = np.sqrt(Exreg**2 + Eyreg**2 + Ezreg**2)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lp_epochs, Exreg*1e3, label='E$_x$ (regular cal)', lw=1.1)
ax.plot(lp_epochs, Eyreg*1e3, label='E$_y$ (regular cal)', lw=1.1)
ax.plot(lp_epochs, Ezreg*1e3, label='E$_z$ (regular cal)', lw=1.1)
ax.plot(lp_epochs, Emagreg*1e3, label='|E| (regular cal)', color = 'purple', lw=1.1)
ax.set_title(f"Frame: {FRAME}, downsample factor: {downsample_factor}")
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.04, -0.02)
ax2 = ax.twinx()
ax2.plot(lp_epochs, U40, label='U40', color='tab:red', linewidth=1.2)
ax2.set_ylabel('U40 (V)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc = 'upper right')
ticks = ax.get_xticks()
ticks = Ef.convert_1970(ticks)
distances = Ef.get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Electric Field (mV/m)')
ax.legend(loc = 'upper left')
ax.grid()

# PLOTTING B FIELD FROM IGRF
Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(vxB_epochs, Bx, label='B$_x$')
ax.plot(vxB_epochs, By, label='B$_y$')
ax.plot(vxB_epochs, Bz, label='B$_z$')
ax.plot(vxB_epochs, Bmag, label='|B|', color='purple')
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.06, -0.025)
ticks = ax.get_xticks()
ticks = Ef.convert_1970(ticks)
distances = Ef.get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Magnetic Field (nT)')
ax.set_title(f'Magnetic Field Components at JUICE in {FRAME} frame')
ax.legend()
ax.grid()

# PLOTTING SPEED FROM SPACECRAFT AND COROTATION
vxtot = vxSC - vxCorot
vytot = vySC - vyCorot
vztot = vzSC - vzCorot

v = np.sqrt(vxtot**2 + vytot**2 + vztot**2)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(vxB_epochs, vxSC, label='v$_x$ from spacecraft', linestyle=':', color='tab:blue')
ax.plot(vxB_epochs, vySC, label='v$_y$ from spacecraft', linestyle=':', color='tab:orange')
ax.plot(vxB_epochs, vzSC, label='v$_z$ from spacecraft', linestyle=':', color='tab:green')
ax.plot(vxB_epochs, vxCorot, label='v$_x$ from corotation', linestyle='--', color='tab:blue')
ax.plot(vxB_epochs, vyCorot, label='v$_y$ from corotation', linestyle='--', color='tab:orange')
ax.plot(vxB_epochs, vzCorot, label='v$_z$ from corotation', linestyle='--', color='tab:green')
ax.plot(vxB_epochs, vxtot, label='v$_x$', color='tab:blue')
ax.plot(vxB_epochs, vytot, label='v$_y$', color='tab:orange')
ax.plot(vxB_epochs, vztot, label='v$_z$', color='tab:green')
ax.plot(vxB_epochs, v, label='|v|', color='purple')
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.06, -0.025)
ticks = ax.get_xticks()
ticks = Ef.convert_1970(ticks)
distances = Ef.get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Velocity (km/s)')
ax.set_title(f'JUICE Spacecraft Velocity Components in {FRAME} frame')
ax.legend(loc = 'upper left')
ax.grid()

# PLOTTING ELECTRIC FIELD FROM -v x B
EvxV = np.sqrt(EvxBx**2 + EvxBy**2 + EvxBz**2)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(vxB_epochs, EvxBx, label='-vxB$_x$')
ax.plot(vxB_epochs, EvxBy, label='-vxB$_y$')
ax.plot(vxB_epochs, EvxBz, label='-vxB$_z$')
ax.plot(vxB_epochs, EvxV, label='|-v x B|', color='purple')
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.06, -0.025)
ticks = ax.get_xticks()
ticks = Ef.convert_1970(ticks)
distances = Ef.get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Electric Field (mV/m)')
ax.set_title(f'Electric Field from -v x B at JUICE in {FRAME} frame')
ax.legend()
ax.grid()

spice.kclear()

print("Finished")

plt.show()