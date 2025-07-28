# Getting distance from THEMIS probes to JUICE spacecraft during the plasmasphere crossing on 20 August 2024

import cdflib
import numpy as np
import datetime as dt
import spiceypy as spice
import matplotlib.pyplot as plt
from Ef_utilities import *

R_E = 6371  # Earth radius

cdfs = []

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

# Getting JUICE state vectors

spice.furnsh("SPICE/JUICE/kernels/mk/juice_ops.tm")
spice.furnsh("SPICE/gsm_frame.tf")

downsample_factor = 50  # Downsample factor for epochs

lp_epochs = np.array([])

for cdf in cdfs:
    lp_epoch = cdf.varget('Epoch')
    lp_epoch = cdflib.cdfepoch.to_datetime(lp_epoch)
    lp_epoch = np.array(lp_epoch, dtype='datetime64[ms]').astype('O')
    lp_epochs = np.concatenate((lp_epochs, lp_epoch))

states = np.zeros((len(lp_epochs[::downsample_factor]), 6))

for i, epoch in enumerate(lp_epochs[::downsample_factor]):
    print(f"Processing epoch {i+1}/{len(lp_epochs[::downsample_factor])}: {epoch.isoformat()}")
    et = spice.str2et(epoch.isoformat())
    state, _ = spice.spkezr('JUICE', et, 'GSM', 'NONE', 'EARTH')
    states[i] = state

# JUICE positions
x = states[:, 0]
y = states[:, 1]
z = states[:, 2]

tha = cdflib.CDF('DATA/tha_l1s_state_20240820000000_20240821000000_cdaweb.cdf')
thd = cdflib.CDF('DATA/thd_l1s_state_20240820000000_20240821000000_cdaweb.cdf')
the = cdflib.CDF('DATA/the_l1s_state_20240820000000_20240821000000_cdaweb.cdf')

tha_pos = tha.varget('tha_pos_gsm')
thd_pos = thd.varget('thd_pos_gsm')
the_pos = the.varget('the_pos_gsm')

tha_epoch = tha.varget('tha_state_time')

tha_epoch = [dt.datetime(1970, 1, 1, 0, 0, 0) + dt.timedelta(seconds=float(epoch)) for epoch in tha_epoch]

xa =  tha_pos[:, 0]
ya =  tha_pos[:, 1]
za =  tha_pos[:, 2]

xd =  thd_pos[:, 0]
yd =  thd_pos[:, 1]
zd =  thd_pos[:, 2]

xe =  the_pos[:, 0]
ye =  the_pos[:, 1]
ze =  the_pos[:, 2]

# Define the time window
start_time = dt.datetime(2024, 8, 20, 19, 0, 0)
end_time = dt.datetime(2024, 8, 21, 0, 0, 0)

# Filter tha_epoch for times within the window
selected_indices = [i for i, t in enumerate(tha_epoch) if start_time <= t < end_time]
selected_times = [tha_epoch[i] for i in selected_indices]

# Prepare arrays for THEMIS and JUICE positions at selected times
themis_a_positions = np.array([[xa[i], ya[i], za[i]] for i in selected_indices])
themis_d_positions = np.array([[xd[i], yd[i], zd[i]] for i in selected_indices])
themis_e_positions = np.array([[xe[i], ye[i], ze[i]] for i in selected_indices])

# For each selected time, find the closest lp_epoch and get JUICE position
juice_positions = []
for t in selected_times:
    # Find index of closest lp_epoch
    idx = np.argmin([abs((t - lp_t).total_seconds()) for lp_t in lp_epochs[::downsample_factor]])
    juice_positions.append([x[idx], y[idx], z[idx]])
juice_positions = np.array(juice_positions)

# Compute distances (in Earth radii)
dist_tha_juice = np.linalg.norm(themis_a_positions - juice_positions, axis=1) / R_E
dist_thd_juice = np.linalg.norm(themis_d_positions - juice_positions, axis=1) / R_E
dist_the_juice = np.linalg.norm(themis_e_positions - juice_positions, axis=1) / R_E

plt.figure(figsize=(12, 6))
plt.plot(selected_times, dist_tha_juice, label='THA-JUICE Distance', marker='o', markersize=2)
plt.plot(selected_times, dist_thd_juice, label='THD-JUICE Distance', marker='o', markersize=2)
plt.plot(selected_times, dist_the_juice, label='THE-JUICE Distance', marker='o', markersize=2)
plt.title('Distance from THEMIS Probes to JUICE')
plt.xlabel('Time')
plt.ylabel('Distance (Earth radii)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

plt.show()

spice.kclear()